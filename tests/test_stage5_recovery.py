import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cases import DIALMED_TRAIN_PATH, load_dialmed_cases  # noqa: E402
from hospital_process import (CAPABILITIES, HospitalSessionState,
                              ProcessHospitalSession)  # noqa: E402
from ipc_runtime import IpcRuntimeState  # noqa: E402
from performance_baseline import FixedModel  # noqa: E402
from tests.test_hospital_process import PositiveModel  # noqa: E402


class SlowDoctor(FixedModel):
    def chat(self, messages):
        if messages[0]['content'].startswith(('你是一名专业的医生', '你是一名医生')):
            time.sleep(0.4)
        return super().chat(messages)


class CrashDoctor(FixedModel):
    def chat(self, messages):
        if messages[0]['content'].startswith(('你是一名专业的医生', '你是一名医生')):
            os._exit(19)
        return super().chat(messages)


class Stage5RecoveryTests(unittest.TestCase):
    @staticmethod
    def _paths(root: Path) -> dict:
        return {
            'golden_path': root / 'golden.json',
            'reflection_path': root / 'reflections.json',
            'communication_path': root / 'communication.jsonl',
            'control_path': root / 'control.jsonl',
            'llm_path': root / 'llm.jsonl',
            'dataset_path': DIALMED_TRAIN_PATH,
        }

    @staticmethod
    def _workers(session):
        return [session.runtime._workers[name][0] for name in CAPABILITIES]

    def _assert_retired(self, workers):
        active = {child.pid for child in mp.active_children()}
        self.assertTrue(all(not worker.is_alive() and worker.exitcode is not None
                            for worker in workers))
        self.assertTrue(all(worker.pid not in active for worker in workers))

    def test_normal_case_transitions_and_clean_close(self):
        case = load_dialmed_cases(DIALMED_TRAIN_PATH, 1)[0]
        with tempfile.TemporaryDirectory() as temporary:
            session = ProcessHospitalSession(
                retrieval_mode='jaccard', verbose=False, training=False, llm_client_factory=FixedModel,
                **self._paths(Path(temporary)),
            )
            workers = self._workers(session)
            try:
                self.assertEqual(session.state, HospitalSessionState.READY)
                self.assertEqual(session.runtime.state, IpcRuntimeState.READY)
                session.run_case(case)
                self.assertEqual(session.state, HospitalSessionState.READY)
                self.assertEqual(session.runtime.state, IpcRuntimeState.READY)
                self.assertIsNone(session.runtime.trace_id)
                for role in CAPABILITIES:
                    snapshot = session.runtime.control_agent(role, 'snapshot')
                    self.assertEqual((snapshot['phase'], snapshot['state'], snapshot['trace_id']),
                                     ('IDLE', 'IDLE', None))
            finally:
                session.close()
                session.close()
            self.assertEqual(session.state, HospitalSessionState.CLOSED)
            self.assertEqual(session.runtime.state, IpcRuntimeState.CLOSED)
            self._assert_retired(workers)

    def test_timeout_propagates_retires_all_and_new_session_runs(self):
        cases = load_dialmed_cases(DIALMED_TRAIN_PATH, 2)
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._paths(Path(temporary))
            session = ProcessHospitalSession(
                retrieval_mode='jaccard', verbose=False, training=False, llm_client_factory=SlowDoctor,
                timeout=0.15, **paths,
            )
            workers = self._workers(session)
            with self.assertRaisesRegex(RuntimeError, '超时'):
                session.run_case(cases[0])
            self.assertEqual(session.state, HospitalSessionState.FAILED)
            self.assertEqual(session.runtime.state, IpcRuntimeState.FAILED)
            self.assertIsNone(session.runtime.trace_id)
            self.assertIn('超时', session.last_error)
            self.assertEqual(session.runtime._workers, {})
            self._assert_retired(workers)
            with self.assertRaisesRegex(RuntimeError, 'FAILED'):
                session.run_case(cases[1])
            session.close()
            with ProcessHospitalSession(
                retrieval_mode='jaccard', verbose=False, training=False, llm_client_factory=FixedModel,
                **paths,
            ) as rebuilt:
                self.assertEqual(rebuilt.run_case(cases[1])['case_id'], cases[1].case_id)

    def test_crash_and_connection_loss_propagate_and_rebuild(self):
        cases = load_dialmed_cases(DIALMED_TRAIN_PATH, 2)
        for mode in ('crash', 'disconnect'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                paths = self._paths(Path(temporary))
                session = ProcessHospitalSession(
                    retrieval_mode='jaccard', verbose=False, training=False,
                    llm_client_factory=CrashDoctor if mode == 'crash' else FixedModel,
                    **paths,
                )
                workers = self._workers(session)
                if mode == 'disconnect':
                    session.runtime._workers['DoctorAgent'][2].close()
                with self.assertRaisesRegex(RuntimeError, '退出|连接中断|管理通信失败'):
                    session.run_case(cases[0])
                self.assertEqual(session.state, HospitalSessionState.FAILED)
                self.assertEqual(session.runtime.state, IpcRuntimeState.FAILED)
                self.assertIsNone(session.runtime.trace_id)
                self._assert_retired(workers)
                session.close()
                with ProcessHospitalSession(
                    retrieval_mode='jaccard', verbose=False, training=False, llm_client_factory=FixedModel,
                    **paths,
                ) as rebuilt:
                    self.assertEqual(rebuilt.run_case(cases[1])['case_id'], cases[1].case_id)

    def test_side_effect_case_is_not_retried_after_later_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / 'one_case.jsonl'
            dataset.write_text(json.dumps({
                'dialog': ['患者：头皮屑伴瘙痒三天'], 'label': ['酮康唑'],
            }, ensure_ascii=False) + '\n', encoding='utf-8')
            paths = self._paths(root)
            paths['dataset_path'] = dataset
            case = load_dialmed_cases(dataset, 1)[0]
            with ProcessHospitalSession(
                retrieval_mode='jaccard', verbose=False, training=True, llm_client_factory=PositiveModel,
                **paths,
            ) as session:
                original = session.runtime.control_agent
                injected = False

                def fail_after_review(name, op, payload=None):
                    nonlocal injected
                    if name == 'ExpertAgent' and op == 'snapshot' and not injected:
                        injected = True
                        raise RuntimeError('故障注入：结果收集失败')
                    return original(name, op, payload)

                with patch.object(session.runtime, 'control_agent', side_effect=fail_after_review):
                    with self.assertRaisesRegex(RuntimeError, '结果收集失败'):
                        session.run_case(case)
                self.assertEqual(session.state, HospitalSessionState.READY)
                self.assertEqual(session.runtime.state, IpcRuntimeState.READY)
                self.assertIn(case.case_id, session.failed_case_ids)
                before = paths['golden_path'].read_bytes()
                self.assertEqual(len(json.loads(before)), 1)
                with self.assertRaisesRegex(RuntimeError, '不可在同一会话重试'):
                    session.run_case(case)
                self.assertEqual(paths['golden_path'].read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
