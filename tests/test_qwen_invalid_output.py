"""Invalid Qwen output is a completed failed case, not a fatal session error."""

import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import main
from tests.test_main import recommendation, supported_golden
from tool.llm_client import TokenUsage


class InvalidQwenOutputTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.dataset = self.root / 'train.txt'
        self.dataset.write_text('\n'.join(json.dumps({
            'dialog': ['患者：头皮屑伴瘙痒三天', '医生：[MASK]'], 'label': ['酮康唑'],
        }, ensure_ascii=False) for _ in range(2)), encoding='utf-8')
        self.experiment = self.root / 'experiment'
        self.argv = ['main.py', '--stage', 'train', '--doctor-backend', 'local-qwen',
                     '--dataset', str(self.dataset), '--experiment-dir', str(self.experiment),
                     '--count', '2']

    def run_cli(self, replies, argv=None):
        calls = []
        responses = iter(replies)

        class FakeQwen:
            last_usage = TokenUsage(10, 5, 15)

            def __init__(self, model_path, max_new_tokens, timeout):
                self.timeout = timeout

            def chat(self, messages):
                calls.append(messages)
                value = next(responses)
                if isinstance(value, Exception):
                    raise value
                return value

        def expert_chat(client, messages):
            client.last_usage = TokenUsage(10, 5, 15)
            return supported_golden()

        with patch('sys.argv', argv or self.argv), redirect_stdout(io.StringIO()), \
                patch('tool.local_qwen_client.LocalQwenClient', side_effect=FakeQwen) as factory, \
                patch('tool.llm_client.LLMClient.chat', autospec=True, side_effect=expert_chat) as expert:
            main.main()
        return calls, factory.call_count, expert.call_count

    def test_failed_case_saved_next_case_succeeds_and_resume_skips_both(self):
        invalid = ['not JSON', recommendation('白名单外药名'),
                   json.dumps({'recommended_drugs': [], 'reasoning_summary': '无效'})]
        calls, loads, expert_calls = self.run_cli(invalid + [recommendation('酮康唑')])
        self.assertEqual((len(calls), loads, expert_calls), (4, 1, 1))
        state = json.loads((self.experiment / 'train/progress.json').read_text())
        self.assertEqual(state['completed_case_ids'], [1, 2])
        failed, succeeded = state['results']
        self.assertEqual(failed['status'], 'doctor_failed')
        self.assertIsNone(failed['recommended_drugs'])
        self.assertFalse(failed['exact_match'])
        self.assertIsNone(failed['reflection'])
        self.assertIsNone(failed['golden_experience'])
        self.assertEqual([row['raw_reply'] for row in failed['attempts']], invalid)
        self.assertTrue(succeeded['exact_match'])
        failures = [json.loads(line) for line in
                    (self.experiment / 'train/doctor_failures.jsonl').read_text().splitlines()]
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]['attempts'], failed['attempts'])
        self.assertEqual([row['case_id'] for row in
                          json.loads((self.experiment / 'golden_experience.json').read_text())], [2])
        self.assertEqual(json.loads((self.experiment / 'reflection_memory.json').read_text()), [])
        usage = [json.loads(line) for line in
                 (self.experiment / 'train/llm_usage.jsonl').read_text().splitlines()]
        totals = [row for row in usage if row['record_type'] == 'case_total']
        self.assertEqual([row['case_id'] for row in totals], [1, 2])
        self.assertEqual(totals[0]['total_tokens'], 45)
        with patch('sys.argv', self.argv + ['--resume']), redirect_stdout(io.StringIO()), \
                patch('tool.local_qwen_client.LocalQwenClient', side_effect=AssertionError('不得重新加载')):
            main.main()
        self.assertEqual(len((self.experiment / 'train/doctor_failures.jsonl').read_text().splitlines()), 1)
        # Simulate a pre-fix checkpoint; the explicit flag accepts only code changes.
        progress_path = self.experiment / 'train/progress.json'
        state['config']['code_sha256']['main.py'] = 'previous-version'
        progress_path.write_text(json.dumps(state))
        with patch('sys.argv', self.argv + ['--resume', '--allow-code-change']), \
                redirect_stdout(io.StringIO()), \
                patch('tool.local_qwen_client.LocalQwenClient', side_effect=AssertionError('不得重新加载')):
            main.main()
        resumed = json.loads(progress_path.read_text())
        self.assertEqual(resumed['results'], state['results'])
        self.assertEqual(resumed['code_changes'][0]['previous_code_sha256']['main.py'], 'previous-version')

    def test_success_on_third_attempt_is_not_recorded_as_failure(self):
        self.run_cli(['bad', 'bad', recommendation('酮康唑'), recommendation('酮康唑')])
        state = json.loads((self.experiment / 'train/progress.json').read_text())
        self.assertTrue(all(row['exact_match'] for row in state['results']))
        self.assertFalse((self.experiment / 'train/doctor_failures.jsonl').exists())

    def test_consecutive_failed_cases_finish_cleanly(self):
        calls, loads, expert_calls = self.run_cli(['bad'] * 6)
        self.assertEqual((len(calls), loads, expert_calls), (6, 1, 0))
        state = json.loads((self.experiment / 'train/progress.json').read_text())
        self.assertEqual(state['completed_case_ids'], [1, 2])
        self.assertTrue(all(row['status'] == 'doctor_failed' for row in state['results']))

    def test_generation_runtime_error_still_propagates(self):
        with self.assertRaisesRegex(RuntimeError, 'generation failed'):
            self.run_cli([RuntimeError('generation failed')])
        state = json.loads((self.experiment / 'train/progress.json').read_text())
        self.assertEqual(state['completed_case_ids'], [])
        self.assertFalse((self.experiment / 'train/doctor_failures.jsonl').exists())

    def test_empty_replies_are_validation_failures(self):
        calls, _, expert_calls = self.run_cli([ValueError('本地 Qwen3 返回空回复')] * 3
                                              + [recommendation('酮康唑')])
        self.assertEqual((len(calls), expert_calls), (4, 1))
        state = json.loads((self.experiment / 'train/progress.json').read_text())
        self.assertEqual(state['results'][0]['status'], 'doctor_failed')
        self.assertTrue(state['results'][1]['exact_match'])


if __name__ == '__main__':
    unittest.main()
