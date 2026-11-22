import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cases import DIALMED_TRAIN_PATH, load_dialmed_cases  # noqa: E402
from hospital_process import (CAPABILITIES, HospitalSessionState,
                              ProcessHospitalSession)  # noqa: E402
from ipc_runtime import IpcRuntimeState  # noqa: E402
from main import HospitalSession  # noqa: E402
from performance_baseline import FixedModel, ROUTE  # noqa: E402
from protocol import MessageType  # noqa: E402


class FailFirstDoctor(FixedModel):
    def __init__(self):
        super().__init__()
        self.failed = 0

    def chat(self, messages):
        if messages[0]['content'].startswith(('你是一名专业的医生', '你是一名医生')) and self.failed < 3:
            self.failed += 1
            self.prompts.append(messages)
            return '{}'
        return super().chat(messages)


class PositiveModel(FixedModel):
    def chat(self, messages):
        self.prompts.append(messages)
        instruction = messages[0]['content']
        if instruction.startswith('你是一名医生'):
            return json.dumps(['酮康唑'], ensure_ascii=False)
        if '你是一名专业的医生' in instruction:
            return json.dumps({
                'recommended_drugs': ['酮康唑'],
                'reasoning_summary': ('关键证据：头皮屑伴瘙痒；疾病判断：考虑头皮屑相关问题，仍需确认；'
                                      '选药依据：酮康唑适应症覆盖头皮屑和瘙痒。'),
            }, ensure_ascii=False)
        return json.dumps({
            'supported': True, 'evidence': [{'drug': '酮康唑',
            'case_evidence': '头皮屑伴瘙痒',
            'drug_evidence': '头皮屑及其相关的脱屑、鳞屑和瘙痒'}],
            'experience': ('当头皮屑伴瘙痒时，应核对适应症并选择针对头皮屑相关症状的药物，'
                           '因为其适应症覆盖脱屑和瘙痒。'),
        }, ensure_ascii=False)


class DoctorOnlyModel(PositiveModel):
    def chat(self, messages):
        if not messages[0]['content'].startswith('你是一名医生'):
            raise AssertionError('测试阶段不应调用专家模型')
        return super().chat(messages)


class ProcessHospitalTests(unittest.TestCase):
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

    def test_test_stage_skips_review_with_optional_transports(self):
        from cases import DialMedCase
        for options in ({'binary_state': False}, {}, {
                'shared_dialog': True, 'incremental_history': True,
                'incremental_medicines': True, 'shared_medicines': True,
                'structured_data': True}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as temporary:
                paths = self._paths(Path(temporary))
                with ProcessHospitalSession(
                    retrieval_mode='jaccard', verbose=False, training=False,
                    llm_client_factory=DoctorOnlyModel, **paths, **options,
                ) as session:
                    for case_id, actual in enumerate(('酮康唑', '依巴斯汀'), 1):
                        outcome = session.run_case(DialMedCase(case_id, '患者：头皮屑伴瘙痒三天', (actual,)))
                        self.assertEqual(outcome['exact_match'], actual == '酮康唑')
                        self.assertIsNone(outcome['golden_experience'])
                        self.assertIsNone(outcome['reflection'])
                        self.assertEqual(session.state, HospitalSessionState.READY)
                    capabilities = {packet.payload.get('capability') for packet in session.trace.packets}
                    self.assertIn('retrieval.history', capabilities)
                    self.assertNotIn('retrieval.medicines', capabilities)
                    self.assertNotIn('expert.review', capabilities)
                    self.assertEqual(session.medicine_delta_stats, [])
                    self.assertEqual(session.runtime.shared_memory_active_bytes, 0)
                    rows = [json.loads(line) for line in paths['llm_path'].read_text().splitlines()]
                    calls = [row for row in rows if row['record_type'] == 'llm_call']
                    self.assertEqual(len(calls), 2)
                    self.assertTrue(all(row['agent'] == 'DoctorAgent' for row in calls))
                    self.assertFalse(paths['golden_path'].exists())
                    self.assertFalse(paths['reflection_path'].exists())

    def test_ten_cases_match_sync_with_four_independent_processes(self):
        cases = load_dialmed_cases(DIALMED_TRAIN_PATH, 10)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sync_paths = self._paths(root / 'sync')
            process_paths = self._paths(root / 'process')
            with patch('main.LLMClient', side_effect=FixedModel):
                sync = HospitalSession(retrieval_mode='jaccard', verbose=False, training=True, **sync_paths)
                try:
                    expected = [sync.run_case(case) for case in cases]
                finally:
                    sync.close()
            process = ProcessHospitalSession(
                retrieval_mode='jaccard', verbose=False, training=True, llm_client_factory=FixedModel,
                **process_paths,
            )
            workers = [process.runtime._workers[role][0] for role in CAPABILITIES]
            try:
                self.assertEqual(len({worker.pid for worker in workers}), 4)
                results = [process.run_case(case) for case in cases]
                self.assertEqual(results, expected)
                self.assertTrue(all(worker.is_alive() for worker in workers))
                self.assertEqual(len(process.trace.message_costs), 10 * len(ROUTE))
                for index, cost in enumerate(process.trace.message_costs):
                    self.assertEqual((cost.sender, cost.receiver, cost.capability), ROUTE[index % len(ROUTE)])
                for case_cost in process.trace.case_costs:
                    packets = [packet for packet in process.trace.packets
                               if packet.head.trace_id == case_cost.trace_id]
                    self.assertEqual(len(packets), 15)
                    self.assertEqual(len({packet.head.message_id for packet in packets}), 15)
                    for index in range(0, 14, 2):
                        incoming, reply = packets[index:index + 2]
                        self.assertIn(incoming.head.message_type,
                                      (MessageType.REQUEST.value, MessageType.RESPONSE.value))
                        self.assertEqual(reply.head.message_type, MessageType.RESPONSE.value)
                        self.assertEqual(reply.head.correlation_id, incoming.head.message_id)
                        self.assertEqual(reply.head.trace_id, incoming.head.trace_id)
                    self.assertEqual(packets[-1].payload['capability'], 'case.complete')
                doctor_packets = [packet for packet in process.trace.packets
                                  if packet.payload.get('capability') == 'doctor.recommend'
                                  and packet.head.source == 'PatientAgent']
                self.assertEqual(len(doctor_packets), 10)
                for packet in doctor_packets:
                    body = json.loads(packet.payload['text'])
                    self.assertNotIn('actual_drugs', body)
                    self.assertNotIn('drug_labels', body)
                prompts = process.runtime.control_agent('DoctorAgent', 'snapshot')['prompts']
                self.assertEqual(len(prompts), 10)
                self.assertTrue(all('真实药物：' not in prompt[1]['content']
                                    and 'actual_drugs' not in prompt[1]['content']
                                    for prompt in prompts))
                self.assertEqual(process.retrieval_statistics['queries'], 0)
                handoffs = [packet for packet in process.trace.packets
                            if ((packet.payload.get('capability') == 'doctor.recommend'
                                 and packet.head.source == 'DoctorAgent')
                                or (packet.payload.get('capability') == 'expert.evaluate'
                                    and packet.head.source == 'ExpertAgent'))]
                self.assertEqual(len(handoffs), 20)
                for packet in handoffs:
                    body = json.loads(packet.payload['text']) if packet.head.source == 'DoctorAgent' else None
                    if body is not None:
                        self.assertEqual(set(body), {'case_id', 'reasoning_summary'})
                expert_inputs = [packet for packet in process.trace.packets
                                 if packet.payload.get('capability') == 'expert.evaluate'
                                 and packet.head.source == 'DoctorAgent']
                self.assertEqual(len(expert_inputs), 10)
                self.assertTrue(all(set(json.loads(packet.payload['text'])) ==
                                    {'case_id', 'reasoning_summary'} for packet in expert_inputs))
                self.assertEqual(len(process.runtime.binary_frame_stats), 20)
                self.assertTrue(all(row['binary_bytes'] > 0 for row in
                                    process.runtime.binary_frame_stats))
                self.assertEqual(len(process.decision_codec_stats), 20)
            finally:
                process.close()
            self.assertTrue(all(not worker.is_alive() for worker in workers))
            for key in ('golden_path', 'reflection_path'):
                left, right = sync_paths[key], process_paths[key]
                self.assertEqual(left.exists(), right.exists())
                if left.exists():
                    self.assertEqual(json.loads(left.read_text(encoding='utf-8')),
                                     json.loads(right.read_text(encoding='utf-8')))

    def test_handler_error_aborts_case_and_next_case_runs(self):
        cases = load_dialmed_cases(DIALMED_TRAIN_PATH, 2)
        with tempfile.TemporaryDirectory() as temporary:
            with ProcessHospitalSession(
                retrieval_mode='jaccard', verbose=False, training=False, llm_client_factory=FailFirstDoctor,
                **self._paths(Path(temporary)),
            ) as session:
                workers = [session.runtime._workers[role][0] for role in CAPABILITIES]
                with self.assertRaisesRegex(RuntimeError, '连续 3 次'):
                    session.run_case(cases[0])
                doctor_input = next(packet for packet in session.trace.packets
                                    if packet.head.source == 'RetrievalAgent'
                                    and packet.payload.get('capability') == 'doctor.recommend')
                doctor_error = next(packet for packet in session.trace.packets
                                    if packet.head.source == 'DoctorAgent'
                                    and packet.head.message_type == MessageType.ERROR.value)
                self.assertEqual(doctor_error.head.trace_id, doctor_input.head.trace_id)
                self.assertEqual(doctor_error.head.correlation_id, doctor_input.head.message_id)
                self.assertIsNone(session.runtime.trace_id)
                self.assertEqual(session.state, HospitalSessionState.READY)
                self.assertEqual(session.runtime.state, IpcRuntimeState.READY)
                self.assertEqual(session.runtime.control_agent('DoctorAgent', 'snapshot')['phase'],
                                 'IDLE')
                self.assertTrue(all(worker.is_alive() for worker in workers))
                result = session.run_case(cases[1])
                self.assertEqual(result['case_id'], cases[1].case_id)
                self.assertEqual(len(session.trace.case_costs), 2)
            self.assertTrue(all(not worker.is_alive() for worker in workers))

    def test_test_stage_freezes_memory_snapshot(self):
        cases = load_dialmed_cases(DIALMED_TRAIN_PATH, 2)
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._paths(Path(temporary))
            paths['golden_path'].write_text(json.dumps([
                {'case_id': 1, 'experience': '旧经验：核对病因和适应症。'},
            ], ensure_ascii=False), encoding='utf-8')
            with ProcessHospitalSession(
                retrieval_mode='jaccard', verbose=False, training=False, llm_client_factory=FixedModel,
                **paths,
            ) as session:
                paths['golden_path'].write_text(json.dumps([
                    {'case_id': 1, 'experience': '外部新经验：核对病因和适应症。'},
                ], ensure_ascii=False), encoding='utf-8')
                session.run_case(cases[1])
                doctor_packets = [packet for packet in session.trace.packets
                                  if packet.payload.get('capability') == 'doctor.recommend'
                                  and packet.head.source == 'RetrievalAgent']
                history = json.loads(doctor_packets[0].payload['text'])['history']
                self.assertNotIn('外部新经验：核对病因和适应症。', history['golden_experiences'])
            self.assertEqual(json.loads(paths['golden_path'].read_text(encoding='utf-8'))[0]['experience'],
                             '外部新经验：核对病因和适应症。')

    def test_training_writes_positive_lesson_without_retrieving_next_case(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._paths(Path(temporary))
            dataset = Path(temporary) / 'two_cases.jsonl'
            dataset.write_text(''.join(json.dumps({
                'dialog': [dialog], 'label': ['酮康唑'],
            }, ensure_ascii=False) + '\n' for dialog in (
                '患者：头皮屑伴瘙痒三天', '患者：头皮屑伴瘙痒四天',
            )), encoding='utf-8')
            paths['dataset_path'] = dataset
            cases = load_dialmed_cases(dataset, 2)
            with ProcessHospitalSession(
                retrieval_mode='embedding', verbose=False, training=True, llm_client_factory=PositiveModel,
                incremental_history=True, shared_dialog=True, structured_data=True,
                **paths,
            ) as session:
                first = session.run_case(cases[0])
                self.assertTrue(first['exact_match'])
                self.assertIsNotNone(first['golden_experience'])
                self.assertEqual(json.loads(paths['golden_path'].read_text(encoding='utf-8')),
                                 [{'case_id': 1, 'experience': first['golden_experience']}])
                session.run_case(cases[1])
                doctor_packets = [packet for packet in session.trace.packets
                                  if packet.payload.get('capability') == 'doctor.recommend'
                                  and packet.head.source == 'PatientAgent']
                self.assertEqual(len(doctor_packets), 2)
                for packet in doctor_packets:
                    body = packet.payload['data']
                    self.assertNotIn('history', body)
                    self.assertNotIn('query_embedding', body)
                    self.assertNotIn('actual_drugs', body)
                self.assertEqual(session.retrieval_statistics['queries'], 0)
                self.assertEqual(session.history_delta_stats, [])
                self.assertNotIn('retrieval.history',
                                 {p.payload.get('capability') for p in session.trace.packets})

    def test_corrupt_binary_returns_correlated_error_and_next_case_runs(self):
        cases = load_dialmed_cases(DIALMED_TRAIN_PATH, 2)
        with tempfile.TemporaryDirectory() as temporary:
            with ProcessHospitalSession(
                retrieval_mode='jaccard', verbose=False, training=True, llm_client_factory=FixedModel,
                **self._paths(Path(temporary)),
            ) as session:
                original = session.runtime.deliver_binary
                with patch.object(session.runtime, 'deliver_binary',
                                  side_effect=lambda packet, binary: original(packet, b'BAD' + binary)):
                    with self.assertRaisesRegex(RuntimeError, '二进制决策状态'):
                        session.run_case(cases[0])
                expert_input = next(packet for packet in session.trace.packets
                                    if packet.payload.get('capability') == 'expert.evaluate'
                                    and packet.head.source == 'DoctorAgent')
                error = next(packet for packet in session.trace.packets
                             if packet.head.message_type == MessageType.ERROR.value
                             and packet.head.source == 'ExpertAgent')
                self.assertEqual(error.head.correlation_id, expert_input.head.message_id)
                self.assertEqual(error.head.trace_id, expert_input.head.trace_id)
                self.assertEqual(session.state, HospitalSessionState.READY)
                self.assertEqual(session.runtime.state, IpcRuntimeState.READY)
                self.assertFalse(self._paths(Path(temporary))['golden_path'].exists())
                self.assertEqual(session.run_case(cases[1])['case_id'], cases[1].case_id)


if __name__ == '__main__':
    unittest.main()
