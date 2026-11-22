"""Offline checks for local Doctor routing and frozen test memories."""

import json
import io
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main
from cases import DIALMED_TEST_PATH, load_dialmed_cases
from tool.llm_client import TokenUsage


class QwenExperimentTests(unittest.TestCase):
    def test_training_uses_local_doctor_and_api_expert(self):
        doctor_calls, expert_calls = [], []

        class FakeQwen:
            last_usage = TokenUsage(100, 50, 150)

            def __init__(self, model_path, max_new_tokens, timeout):
                self.timeout = timeout
                self.max_new_tokens = max_new_tokens

            def chat(self, messages):
                doctor_calls.append(messages)
                return json.dumps({
                    'recommended_drugs': ['酮康唑'],
                    'reasoning_summary': '关键证据：头皮屑伴瘙痒；疾病判断：考虑头皮屑相关问题，仍需确认；'
                                         '选药依据：酮康唑适应症覆盖头皮屑和瘙痒。',
                }, ensure_ascii=False)

        def expert_chat(client, messages):
            expert_calls.append(messages)
            client.last_usage = TokenUsage(100, 50, 150)
            return json.dumps({
                'supported': True,
                'evidence': [{'drug': '酮康唑', 'case_evidence': '头皮屑伴瘙痒',
                              'drug_evidence': '头皮屑及其相关的脱屑、鳞屑和瘙痒'}],
                'experience': '当头皮屑伴瘙痒时，应核对适应症并选择针对头皮屑相关症状的药物，'
                              '因为其适应症覆盖脱屑和瘙痒。',
            }, ensure_ascii=False)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / 'train.txt'
            dataset.write_text(json.dumps({'dialog': ['患者：头皮屑伴瘙痒三天', '医生：[MASK]'],
                                           'label': ['酮康唑']}, ensure_ascii=False))
            experiment = root / 'experiment'
            argv = ['main.py', '--stage', 'train', '--doctor-backend', 'local-qwen',
                    '--doctor-max-new-tokens', '512',
                    '--experiment-dir', str(experiment), '--dataset', str(dataset), '--count', '1']
            with patch.object(sys, 'argv', argv), redirect_stdout(io.StringIO()), \
                    patch('tool.local_qwen_client.LocalQwenClient', FakeQwen), \
                    patch('tool.llm_client.LLMClient.chat', expert_chat), \
                    patch('agent.retrieval.RetrievalAgent.retrieve_history',
                          side_effect=AssertionError('训练不能检索经验')), \
                    patch('embedding.encoder.ApiBackend.encode',
                          side_effect=AssertionError('训练不能生成经验查询或索引向量')):
                main.main()
            self.assertEqual(len(doctor_calls), 1)
            self.assertTrue(expert_calls)
            self.assertIn('reasoning_summary', doctor_calls[0][0]['content'])
            state = json.loads((experiment / 'train/progress.json').read_text())
            self.assertEqual(state['completed_case_ids'], [1])
            self.assertEqual(state['config']['doctor_generation']['max_new_tokens'], 512)
            self.assertIsNone(state['config']['memory_dir'])
            self.assertTrue(json.loads((experiment / 'golden_experience.json').read_text()))
            self.assertNotIn('相关历史经验', doctor_calls[0][1]['content'])
            self.assertFalse((experiment / 'semantic-memory.sqlite').exists())
            with patch.object(sys, 'argv', argv + ['--resume']), redirect_stdout(io.StringIO()), \
                    patch('tool.local_qwen_client.LocalQwenClient',
                          side_effect=AssertionError('已完成的续训不应重新加载模型')):
                main.main()

    def test_cli_routes_doctor_locally_and_reuses_frozen_history_on_resume(self):
        case = load_dialmed_cases(DIALMED_TEST_PATH, 1)[0]
        calls = []

        class FakeQwen:
            last_usage = TokenUsage(100, 10, 110)

            def __init__(self, model_path, max_new_tokens, timeout):
                self.timeout = timeout
                self.asserted_max_new_tokens = max_new_tokens

            def chat(self, messages):
                calls.append(messages)
                phrase = json.loads(case.patient_input)[0][:24]
                return json.dumps({
                    'recommended_drugs': list(case.drug_labels),
                    'reasoning_summary': f'关键证据：{phrase}；疾病判断：病因待明确；'
                                         '选药依据：结合症状核对药物适应症。',
                }, ensure_ascii=False)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'history'
            source.mkdir()
            names = ('golden_experience.json', 'reflection_memory.json')
            original = {}
            for name in names:
                rows = json.loads((main.GOLDEN_EXPERIENCE_PATH.parent / name).read_bytes())
                # One real record of each kind fits within the normal top-3 retrieval.
                original[name] = json.dumps(rows[:1], ensure_ascii=False).encode()
                (source / name).write_bytes(original[name])
            experiment = root / 'experiment'
            argv = ['main.py', '--stage', 'test', '--doctor-backend', 'local-qwen',
                    '--memory-dir', str(source), '--experiment-dir', str(experiment),
                    '--count', '1']
            with patch.object(sys, 'argv', argv), \
                    redirect_stdout(io.StringIO()), \
                    patch('tool.local_qwen_client.LocalQwenClient', FakeQwen), \
                    patch('embedding.encoder.ApiBackend.encode',
                          side_effect=lambda texts: [[1.0, 0.0] for _ in texts]), \
                    patch('tool.llm_client.LLMClient.chat', side_effect=AssertionError('Expert must not call API in test')):
                main.main()
            self.assertEqual(len(calls), 1)
            self.assertIn('药物名称必须与白名单完全一致', calls[0][0]['content'])
            self.assertIn('相关历史经验：', calls[0][1]['content'])
            history = json.loads(calls[0][1]['content'].split('相关历史经验：', 1)[1])
            self.assertTrue(history['golden_experiences'])
            self.assertTrue(history['reflections'])
            state = json.loads((experiment / 'test/progress.json').read_text())
            self.assertEqual(state['completed_case_ids'], [case.case_id])
            self.assertTrue(state['results'][0]['exact_match'])
            self.assertEqual(state['config']['chat_endpoint'], 'local')
            self.assertEqual(state['config']['doctor_generation']['max_new_tokens'], 256)
            usage = [json.loads(line) for line in (experiment / 'test/llm_usage.jsonl').read_text().splitlines()]
            self.assertTrue(usage)
            for name in names:
                self.assertEqual((source / name).read_bytes(), original[name])
                self.assertEqual((experiment / name).read_bytes(), original[name])
                # Later training changes must not alter the frozen test snapshot.
                (source / name).write_text('[]')
            with patch.object(sys, 'argv', argv + ['--resume']), \
                    redirect_stdout(io.StringIO()), \
                    patch('tool.local_qwen_client.LocalQwenClient', side_effect=AssertionError('No model reload on completed resume')):
                main.main()

    def test_refuses_to_overwrite_different_memories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, target = root / 'source', root / 'target'
            source.mkdir()
            target.mkdir()
            for name in ('golden_experience.json', 'reflection_memory.json'):
                (source / name).write_text('[]')
                (target / name).write_text('[{"case_id": 1}]')
            with self.assertRaises(ValueError):
                main.prepare_test_memories(source, target, resume=False)
            self.assertEqual((target / 'golden_experience.json').read_text(), '[{"case_id": 1}]')

    def test_local_process_mode_is_rejected_before_loading(self):
        with self.assertRaisesRegex(ValueError, 'sync'):
            main.run_cases([], doctor_backend='local-qwen', runtime_mode='process')


if __name__ == '__main__':
    unittest.main()
