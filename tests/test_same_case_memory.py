"""Offline verification of same-case replay; never load a real model."""
import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent.doctor_Qwen import DoctorAgent
from cases import DialMedCase
from experiment import same_case_memory as replay
from tool.tracing import Trace


class FakeClient:
    def __init__(self, *args, **kwargs):
        self.calls = []
        self.last_usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)

    def chat(self, messages):
        self.calls.append(messages)
        drug = '奥美拉唑' if '相关历史经验：' in messages[1]['content'] else '莫沙必利'
        return json.dumps({
            'recommended_drugs': [drug],
            'reasoning_summary': f'关键证据：患者胃痛反酸；疾病判断：考虑胃炎；选药依据：{drug}用于改善胃部不适症状。',
        }, ensure_ascii=False)


class SameCaseMemoryTests(unittest.TestCase):
    def test_prompt_and_retry_parity_with_original_test_agent(self):
        medicines = SimpleNamespace(names=('奥美拉唑', '莫沙必利'))
        case = DialMedCase(7, json.dumps(['患者：胃痛反酸', '医生：[MASK]'], ensure_ascii=False),
                           ('奥美拉唑',))
        history = {'golden_experiences': ['胃痛反酸时参考经验。'], 'reflections': []}
        for selected in (history, replay.empty_history()):
            with self.subTest(history=selected):
                class RetryClient(FakeClient):
                    def chat(self, messages):
                        reply = super().chat(messages)
                        return 'not json' if len(self.calls) == 1 else reply

                original_client, new_client = RetryClient(), RetryClient()
                trace = Trace(verbose=False)
                trace.message_costs.append(SimpleNamespace(step=1))
                original = DoctorAgent(trace, medicines, lambda: original_client, training=False)
                original_result = json.loads(original.recommend(json.dumps({
                    'case_id': 7, 'count': 1, 'patient_dialog': case.patient_input,
                    'history': selected,
                }, ensure_ascii=False)))
                result = replay.predict(case, selected, new_client, medicines)
                self.assertEqual(original_client.calls, new_client.calls)
                self.assertEqual(len(new_client.calls), 2)
                self.assertEqual(original_result['recommended_drugs'], result['recommended_drugs'])
                self.assertNotIn('actual_drugs', str(new_client.calls))

    def test_memory_ids_are_not_array_positions(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / 'golden_experience.json').write_text(json.dumps([
                {'case_id': 7, 'experience': '第七条'},
                {'case_id': 2, 'experience': '第二条'},
            ]))
            (directory / 'reflection_memory.json').write_text(json.dumps([
                {'case_id': 7, 'reflection': '第七条反思'},
            ]))
            _, lookup = replay.load_memories(directory, {1, 2, 7}, 'both')
            self.assertNotIn(1, lookup)
            self.assertEqual(lookup[2]['golden_experiences'], ['第二条'])
            self.assertEqual(replay.memory_type(lookup[7]), 'both')
            _, lookup = replay.load_memories(directory, {1, 2, 7}, 'reflection')
            self.assertEqual(lookup[7]['golden_experiences'], [])
            self.assertNotIn(2, lookup)
            with self.assertRaises(ValueError):
                replay.load_memories(directory, {1, 2}, 'both')
            (directory / 'reflection_memory.json').write_text('')
            with self.assertRaisesRegex(ValueError, 'reflection_memory.json'):
                replay.load_memories(directory, {1, 2, 7}, 'both')

    def test_paired_run_resume_and_frozen_memories(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'memory'
            source.mkdir()
            dataset = root / 'train.jsonl'
            dataset.write_text('\n'.join(json.dumps({
                'dialog': ['患者：胃痛反酸', '医生：[MASK]'], 'label': ['奥美拉唑'],
            }, ensure_ascii=False) for _ in range(3)), encoding='utf-8')
            (source / 'golden_experience.json').write_text(json.dumps([
                {'case_id': 2, 'experience': '胃痛反酸时参考奥美拉唑。'},
            ]), encoding='utf-8')
            (source / 'reflection_memory.json').write_text(json.dumps([
                {'case_id': 1, 'reflection': '胃痛反酸时核查抑酸治疗。'},
            ]), encoding='utf-8')
            output = root / 'result'
            args = ['--dataset', str(dataset), '--memory-dir', str(source),
                    '--experiment-dir', str(output)]
            client = FakeClient()
            with patch.object(replay, 'LocalQwenClient', return_value=client), redirect_stdout(io.StringIO()):
                replay.main(args)
            self.assertEqual(len(client.calls), 5)  # 2 paired cases + 1 uncovered case.
            state = replay.read_json(output / 'progress.json')
            self.assertEqual(state['completed_case_ids'], [1, 2, 3])
            self.assertEqual([r['memory_type'] for r in state['results']],
                             ['reflection', 'golden', 'none'])
            self.assertTrue(state['results'][2]['baseline_reused'])
            metrics = replay.read_json(output / 'metrics.json')
            self.assertEqual(metrics['cases_with_memory']['transitions']['wrong_to_right'], [1, 2])
            self.assertEqual(metrics['cases_with_memory']['net_correct_gain'], 2)
            with patch.object(replay, 'LocalQwenClient', side_effect=AssertionError('no reload')), \
                    redirect_stdout(io.StringIO()):
                replay.main(args + ['--resume'])
            # Simulate interruption after the first paired case was committed.
            state['results'] = state['results'][:1]
            state['completed_case_ids'] = [1]
            replay.atomic_json(output / 'progress.json', state)
            resumed = FakeClient()
            with patch.object(replay, 'LocalQwenClient', return_value=resumed), redirect_stdout(io.StringIO()):
                replay.main(args + ['--resume'])
            self.assertEqual(len(resumed.calls), 3)
            self.assertEqual(replay.read_json(output / 'progress.json')['completed_case_ids'], [1, 2, 3])
            (source / 'golden_experience.json').write_text('[]')
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                replay.main(args + ['--resume'])


if __name__ == '__main__':
    unittest.main()
