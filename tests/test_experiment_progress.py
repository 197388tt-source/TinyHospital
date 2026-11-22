import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from tool.api_retry import request_json
from tool.experiment_progress import ExperimentProgress


class ProgressTests(unittest.TestCase):
    def test_explicit_code_change_resume_preserves_results_and_rejects_data_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'train').mkdir()
            for name in ('golden_experience.json', 'reflection_memory.json'):
                (root / name).write_text('[]')
            old = {'dataset': 'a', 'code_sha256': {'main.py': 'old'}}
            new = {'dataset': 'a', 'code_sha256': {'main.py': 'new'}}
            p = ExperimentProgress(root, 'train', old, False)
            p.commit({'case_id': 1})
            p.close()
            with self.assertRaises(ValueError):
                ExperimentProgress(root, 'train', new, True)
            with self.assertRaises(ValueError):
                ExperimentProgress(root, 'train', {**new, 'dataset': 'b'}, True,
                                   allow_code_change=True)
            p = ExperimentProgress(root, 'train', new, True, allow_code_change=True)
            self.assertEqual(p.state['results'], [{'case_id': 1}])
            self.assertEqual(p.state['config'], new)
            self.assertEqual(p.state['code_changes'][0]['previous_code_sha256'], old['code_sha256'])
            self.assertEqual(p.state['code_changes'][0]['completed_case_ids'], [1])
            p.close()
            p = ExperimentProgress(root, 'train', new, True)
            self.assertEqual(p.state['completed_case_ids'], [1])
            self.assertEqual(len(p.state['code_changes']), 1)
            p.close()

    def test_restart_restores_committed_memory_and_skips_committed_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'train').mkdir()
            for name in ('golden_experience.json', 'reflection_memory.json'):
                (root/name).write_text('[]')
            p = ExperimentProgress(root, 'train', {'dataset': 'a'}, False)
            memory = root/'reflection_memory.json'
            memory.write_text('[{"case_id":1,"reflection":"committed"}]')
            p.commit({'case_id': 1, 'recommended_drugs': ['a']})
            memory.write_text('[{"case_id":2,"reflection":"uncommitted"}]')
            p.close()
            with self.assertRaises(ValueError):
                ExperimentProgress(root, 'train', {'dataset': 'b'}, True)
            p = ExperimentProgress(root, 'train', {'dataset': 'a'}, True)
            self.assertEqual(p.state['completed_case_ids'], [1])
            self.assertEqual(json.loads(memory.read_text())[0]['case_id'], 1)
            p.close()
            with self.assertRaises(ValueError):
                ExperimentProgress(root, 'train', {'dataset': 'a'}, False)

    def test_retry_transient_but_not_auth(self):
        with patch.dict('os.environ', {'TINY_HOSPITAL_API_RETRIES': '2'}), patch('tool.api_retry.time.sleep'):
            with patch('tool.llm_client.urlopen', side_effect=[URLError('offline'), io.StringIO('{"ok":true}')]) as opener:
                self.assertEqual(request_json(opener, object(), 1), {'ok': True})
                self.assertEqual(opener.call_count, 2)
            with patch('tool.llm_client.urlopen', side_effect=HTTPError('url',401,'denied',{},None)) as opener:
                with self.assertRaises(HTTPError):
                    request_json(opener, object(), 1)
                self.assertEqual(opener.call_count, 1)
            with patch('tool.llm_client.urlopen', side_effect=TimeoutError()) as opener:
                with self.assertRaises(TimeoutError):
                    request_json(opener, object(), 1)
                self.assertEqual(opener.call_count, 3)

    def test_cli_resume_filters_completed_case(self):
        import main
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            args = ['main.py','--count','2','--retrieval-mode','none','--experiment-dir',tmp]
            def fake_run(cases, **kwargs):
                self.assertEqual([c.case_id for c in cases], [1, 2])
                kwargs['on_case_complete']({'case_id': 1})
                raise TimeoutError('interrupted')
            with patch('sys.argv', args), patch('main.run_cases', side_effect=fake_run):
                with self.assertRaises(TimeoutError):
                    main.main()
            with patch('sys.argv', args + ['--resume']), patch('main.run_cases') as run:
                main.main()
                self.assertEqual([c.case_id for c in run.call_args.args[0]], [2])
