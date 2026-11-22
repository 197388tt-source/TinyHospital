import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from history.memory import CaseMemoryStore


class CaseMemoryStoreTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.golden = root / 'golden_experience.json'
        self.reflections = root / 'reflection_memory.json'
        self.dataset = root / 'train.txt'

    def write_dataset(self, *dialogs):
        self.dataset.write_text(''.join(
            json.dumps({'dialog': [dialog], 'label': ['隐藏标签']}, ensure_ascii=False) + '\n'
            for dialog in dialogs
        ), encoding='utf-8')

    @staticmethod
    def write(path, rows):
        path.write_text(json.dumps(rows, ensure_ascii=False), encoding='utf-8')

    def store(self, training=True):
        return CaseMemoryStore(self.golden, self.reflections, training, self.dataset, retrieval_mode='jaccard')

    def test_positive_length_limit_matches_expert_and_survives_reload(self):
        self.write_dataset('头皮屑伴瘙痒三天')
        store = self.store()
        lesson = '字' * 200
        store.save_success(1, lesson)
        self.assertEqual(self.store(training=False)._golden[1], lesson)
        with self.assertRaises(ValueError):
            store.save_success(1, lesson + '字')

    def test_golden_retrieval_needs_no_correct_case_file(self):
        self.write_dataset('头皮屑伴瘙痒三天', '头痛一天')
        lesson = '当头皮屑伴瘙痒时，应核对适应症并选择与头皮症状匹配的药物。'
        store = self.store()
        store.save_success(1, lesson)
        self.assertEqual(json.loads(self.golden.read_text(encoding='utf-8')),
                         [{'case_id': 1, 'experience': lesson}])
        self.assertFalse((self.golden.parent / 'history_memory.json').exists())
        found = store.retrieve('头皮屑伴瘙痒四天', 2, str(self.dataset))
        self.assertEqual(found['golden_experiences'][0]['experience'], lesson)
        self.assertEqual(found['golden_experiences'][0]['summary'], '头皮屑伴瘙痒三天；病因待确认')
        self.assertNotIn('actual_drugs', found['golden_experiences'][0])
        self.assertEqual(store.retrieve('头皮屑伴瘙痒三天', 1, str(self.dataset)),
                         {'golden_experiences': [], 'reflections': []})

    def test_training_refreshes_and_test_freezes(self):
        self.write_dataset('发热 咳嗽 三天', '头痛 一天')
        self.write(self.golden, [{'case_id': 1, 'experience': '旧经验'}])
        train, test = self.store(), self.store(training=False)
        query = '发热 咳嗽 四天'
        self.write(self.golden, [{'case_id': 1, 'experience': '新经验'}])
        self.assertEqual(train.retrieve(query, 2, str(self.dataset))['golden_experiences'][0]['experience'],
                         '新经验')
        self.assertEqual(test.retrieve(query, 2, str(self.dataset))['golden_experiences'][0]['experience'],
                         '旧经验')
        test.save_success(1, '禁止写入')
        test.save_reflection(2, '禁止写入')
        self.assertEqual(json.loads(self.golden.read_text(encoding='utf-8'))[0]['experience'],
                         '新经验')
        self.assertFalse(self.reflections.exists())

    def test_only_approved_success_is_stored_and_stale_reflection_removed(self):
        self.write_dataset('足部瘙痒三天')
        store = self.store()
        store.save_reflection(1, '应核对适应症')
        store.save_success(1, None)
        self.assertEqual(json.loads(self.reflections.read_text(encoding='utf-8')), [])
        self.assertFalse(self.golden.exists())
        store.save_success(1, '当足部瘙痒时，应核对病因与适应症。')
        self.assertEqual(len(json.loads(self.golden.read_text(encoding='utf-8'))), 1)
        store.save_success(1, None)
        self.assertEqual(json.loads(self.golden.read_text(encoding='utf-8')), [])

    def test_error_replaces_golden_and_keeps_two_fields(self):
        self.write_dataset('足部瘙痒三天')
        store = self.store()
        store.save_success(1, '当足部瘙痒时，应核对病因与适应症。')
        store.save_reflection(1, '应核对适应症')
        self.assertEqual(json.loads(self.golden.read_text(encoding='utf-8')), [])
        row = json.loads(self.reflections.read_text(encoding='utf-8'))[0]
        self.assertEqual(row, {'case_id': 1, 'reflection': '应核对适应症'})
        found = store.retrieve('足部瘙痒四天', 2, str(self.dataset))['reflections'][0]
        self.assertEqual(found['summary'], '足部瘙痒三天；病因待确认')
        self.assertNotIn('actual_drugs', found)
        self.assertNotIn('correct_drugs', found)
        self.assertNotIn('incorrect_recommended_drugs', found)

    def test_global_rank_deduplication_and_statistics(self):
        self.write_dataset('发热 咳嗽 胸痛', '发热 咳嗽', '咳嗽 胸痛',
                           '脚趾骨折', '发热 咳嗽')
        self.write(self.golden, [
            {'case_id': 1, 'experience': '当前病例不应返回'},
            {'case_id': 2, 'experience': '发热咳嗽时核对病因'},
            {'case_id': 4, 'experience': '无关经验'},
        ])
        self.write(self.reflections, [
            {'case_id': 3, 'reflection': '核对病因'},
            {'case_id': 5, 'reflection': '重复对话'},
        ])
        store = self.store()
        result = store.retrieve('发热 咳嗽 胸痛', 1, str(self.dataset), limit=2,
                                min_similarity=0.3)
        self.assertEqual(sum(map(len, result.values())), 2)
        self.assertEqual(len(result['golden_experiences']), 1)
        self.assertEqual(len(result['reflections']), 1)
        self.assertEqual(store.last_retrieval_stats['similarities'], [0.6667, 0.6667])
        self.assertEqual(store.retrieval_statistics['hit_rate'], 1.0)
        self.assertGreaterEqual(store.last_retrieval_stats['latency_ms'], 0)
        self.assertEqual(store.retrieve('完全无关的话题', 6, str(self.dataset)),
                         {'golden_experiences': [], 'reflections': []})
        self.assertEqual(store.retrieval_statistics['hit_rate'], 0.5)

    def test_reflection_length_boundary_roundtrip(self):
        self.write_dataset('头皮屑伴瘙痒三天')
        store = self.store()
        reflection = '核对病例条件' * 41 + '再核实。'
        self.assertEqual(len(reflection), 250)
        store.save_reflection(1, reflection)
        reloaded = self.store()
        self.assertEqual(reloaded._reflections[0]['reflection'], reflection)
        for invalid in ('', '   ', reflection + '字'):
            with self.subTest(length=len(invalid)):
                with self.assertRaises(ValueError):
                    store.save_reflection(1, invalid)
        self.assertEqual(json.loads(self.reflections.read_text())[0]['reflection'], reflection)

    def test_legacy_reflection_without_full_dialog_uses_dataset(self):
        self.write_dataset('持续发热咳嗽三天')
        self.write(self.reflections, [{
            'reflection_id': 7, 'case_id': 1, 'summary': '咽炎',
            'recommended_drugs': ['药甲'], 'actual_drugs': ['药乙'],
            'lesson': '应核对药物适应症',
        }])
        found = self.store().retrieve('发热咳嗽四天', 2, str(self.dataset))['reflections'][0]
        self.assertIn('持续发热咳嗽三天', found['dialog_excerpt'])
        self.assertEqual(found['reflection'], '应核对药物适应症')
        store = self.store()
        store.save_reflection(1, '更新后的反思')
        self.assertEqual(json.loads(self.reflections.read_text(encoding='utf-8')),
                         [{'case_id': 1, 'reflection': '更新后的反思'}])


if __name__ == '__main__':
    unittest.main()
