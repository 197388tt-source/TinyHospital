"""Stage 8: only retrieval.medicines replies become incremental."""

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cases import DIALMED_TRAIN_PATH, load_dialmed_cases  # noqa: E402
from hospital_process import (CAPABILITIES, HospitalSessionState,
                              ProcessHospitalSession, _restore_medicine_response)  # noqa: E402
from ipc_runtime import IpcRuntimeState  # noqa: E402
from performance_baseline import FixedModel  # noqa: E402
from protocol import AgentPacket, Delivery, MessageType  # noqa: E402
from tool.medicine_store import MedicineKnowledgeBase  # noqa: E402


class MissingIntroductionModel(FixedModel):
    def __init__(self):
        super().__init__()
        self.names = ('感冒灵',) + tuple(name for name in self.names if name != '感冒灵')


def paths(root, dataset=DIALMED_TRAIN_PATH):
    return {
        'golden_path': root / 'golden.json',
        'reflection_path': root / 'reflection.json',
        'communication_path': root / 'communication.jsonl',
        'control_path': root / 'control.jsonl',
        'llm_path': root / 'llm.jsonl',
        'dataset_path': dataset,
    }


def packets(session, capability, source):
    return [packet for packet in session.trace.packets
            if packet.payload.get('capability') == capability
            and packet.head.source == source]


class IncrementalMedicinesTests(unittest.TestCase):
    def test_ten_cases_and_all_optional_modes_match_full_input(self):
        cases = load_dialmed_cases(DIALMED_TRAIN_PATH, 10)
        # Guarantee a successful case independently of changes to the real dataset.
        cases[0] = replace(cases[0], drug_labels=tuple(FixedModel().names[:cases[0].drug_count]))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with ProcessHospitalSession(
                retrieval_mode='jaccard', verbose=False, training=True, llm_client_factory=FixedModel,
                **paths(root / 'full'),
            ) as full:
                expected = [full.run_case(case) for case in cases]
                full_inputs = [p.payload['text'] for p in packets(
                    full, 'expert.review', 'RetrievalAgent')]
                full_replies = [p.payload['text'] for p in packets(
                    full, 'retrieval.medicines', 'RetrievalAgent')]
                doctor_inputs = [p.payload['text'] for p in packets(
                    full, 'doctor.recommend', 'PatientAgent')]
                self.assertEqual(len(doctor_inputs), 10)
            for label, options in (
                ('delta', {'incremental_medicines': True}),
                ('combined', {'incremental_medicines': True,
                              'incremental_history': True, 'shared_dialog': True}),
            ):
                with self.subTest(mode=label):
                    with ProcessHospitalSession(
                        retrieval_mode='jaccard', verbose=False, training=True, llm_client_factory=FixedModel,
                        **options, **paths(root / label),
                    ) as session:
                        workers = [session.runtime._workers[role][0] for role in CAPABILITIES]
                        actual = [session.run_case(case) for case in cases]
                        self.assertEqual(actual, expected)
                        self.assertEqual([p.payload['text'] for p in packets(
                            session, 'expert.review', 'RetrievalAgent')], full_inputs)
                        self.assertEqual([p.payload['text'] for p in packets(
                            session, 'doctor.recommend', 'PatientAgent')], doctor_inputs)
                        self.assertTrue(all('actual_drugs' not in json.loads(text)
                                            and 'drug_labels' not in json.loads(text)
                                            for text in doctor_inputs))
                        replies = [p.payload['text'] for p in packets(
                            session, 'retrieval.medicines', 'RetrievalAgent')]
                        self.assertEqual(len(replies), 10)
                        self.assertTrue(all(set(json.loads(text)) ==
                                            {'case_id', 'medicine_introductions'}
                                            for text in replies))
                        self.assertTrue(all(len(new.encode('utf-8')) < len(old.encode('utf-8'))
                                            for old, new in zip(full_replies, replies)))
                        self.assertIsNone(session._medicine_context)
                        self.assertIsNone(session._history_context)
                        self.assertEqual(len(session.medicine_delta_stats), 10)
                        self.assertEqual(session.runtime.shared_memory_active_bytes, 0)
                        self.assertTrue(all(worker.is_alive() for worker in workers))
                        prompts = session.runtime.control_agent('DoctorAgent', 'snapshot')['prompts']
                        self.assertTrue(all('actual_drugs' not in prompt[1]['content']
                                            and 'drug_labels' not in prompt[1]['content']
                                            for prompt in prompts))
                        if options.get('incremental_history'):
                            self.assertEqual(len(session.history_delta_stats), 0)
                            self.assertTrue(any(row['mode'] == 'shared' for row in
                                                session.runtime.transport_frame_stats))
                    self.assertTrue(all(not worker.is_alive() for worker in workers))
            self.assertTrue(any(item['exact_match'] for item in expected))
            self.assertTrue(any(not item['exact_match'] for item in expected))
            self.assertTrue(any(set(item['recommended_drugs']) & set(item['actual_drugs'])
                                for item in expected))

    def test_correct_wrong_overlap_and_missing_introduction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / 'cases.jsonl'
            labels = (["酮康唑"], ["依巴斯汀"], ["酮康唑", "黄连素"], ["黄连素"])
            dataset.write_text(''.join(json.dumps({
                'dialog': [f'患者：头皮屑伴瘙痒{index}天'], 'label': label,
            }, ensure_ascii=False) + '\n' for index, label in enumerate(labels, 1)),
                encoding='utf-8')
            cases = load_dialmed_cases(dataset, 4)
            self.assertEqual(MedicineKnowledgeBase().introductions_for(['感冒灵'])['感冒灵'],
                             {'说明': '药物说明文件中暂无该药物的记录'})
            for mode in (False, True):
                with ProcessHospitalSession(
                    retrieval_mode='jaccard', verbose=False, training=True, llm_client_factory=FixedModel,
                    incremental_medicines=mode,
                    **paths(root / f'fixed_{mode}', dataset),
                ) as session:
                    outcomes = [session.run_case(case) for case in cases[:3]]
                    self.assertTrue(outcomes[0]['exact_match'])
                    self.assertFalse(outcomes[1]['exact_match'])
                    self.assertEqual(set(outcomes[2]['recommended_drugs']) &
                                     set(outcomes[2]['actual_drugs']), {'酮康唑'})
                    texts = [p.payload['text'] for p in packets(
                        session, 'expert.review', 'RetrievalAgent')]
                    if mode:
                        delta_outcomes, delta_texts = outcomes, texts
                    else:
                        full_outcomes, full_texts = outcomes, texts
            self.assertEqual(delta_outcomes, full_outcomes)
            self.assertEqual(delta_texts, full_texts)
            for mode in (False, True):
                with ProcessHospitalSession(
                    retrieval_mode='jaccard', verbose=False, training=True,
                    llm_client_factory=MissingIntroductionModel,
                    incremental_medicines=mode,
                    **paths(root / f'missing_{mode}', dataset),
                ) as session:
                    outcome = session.run_case(cases[3])
                    expert_input = packets(session, 'expert.review', 'RetrievalAgent')[0]
                    introductions = json.loads(expert_input.payload['text'])['medicine_introductions']
                    self.assertEqual(introductions['感冒灵'],
                                     {'说明': '药物说明文件中暂无该药物的记录'})
                    if mode:
                        delta_missing = (outcome, expert_input.payload['text'])
                    else:
                        full_missing = (outcome, expert_input.payload['text'])
            self.assertEqual(delta_missing, full_missing)

    def test_wrong_case_reply_aborts_and_next_case_runs(self):
        cases = load_dialmed_cases(DIALMED_TRAIN_PATH, 2)
        with tempfile.TemporaryDirectory() as temporary:
            with ProcessHospitalSession(
                retrieval_mode='jaccard', verbose=False, training=True, llm_client_factory=FixedModel,
                incremental_medicines=True, **paths(Path(temporary)),
            ) as session:
                original = session.runtime.deliver
                injected = False

                def corrupt(packet):
                    nonlocal injected
                    delivery = original(packet)
                    if packet.payload.get('capability') == 'retrieval.medicines' and not injected:
                        injected = True
                        body = json.loads(delivery.output)
                        body['case_id'] = 9999
                        text = json.dumps(body, ensure_ascii=False)
                        response = AgentPacket(delivery.packet.head, {
                            'capability': 'retrieval.medicines', 'text': text,
                        })
                        session.trace.packets[-1] = response
                        return Delivery(response, text)
                    return delivery

                with patch.object(session.runtime, 'deliver', side_effect=corrupt):
                    with self.assertRaisesRegex(ValueError, '病例编号不匹配'):
                        session.run_case(cases[0])
                self.assertIsNone(session._medicine_context)
                self.assertIsNone(session.runtime.trace_id)
                self.assertEqual(session.state, HospitalSessionState.READY)
                self.assertEqual(session.runtime.state, IpcRuntimeState.READY)
                for role in CAPABILITIES:
                    state = session.runtime.control_agent(role, 'snapshot')
                    self.assertEqual((state['phase'], state['trace_id']), ('IDLE', None))
                self.assertEqual(session.run_case(cases[1])['case_id'], cases[1].case_id)
                self.assertIsNone(session._medicine_context)

    def test_worker_exit_retires_session_and_rebuilds_without_old_context(self):
        cases = load_dialmed_cases(DIALMED_TRAIN_PATH, 2)
        with tempfile.TemporaryDirectory() as temporary:
            case_paths = paths(Path(temporary))
            session = ProcessHospitalSession(
                retrieval_mode='jaccard', verbose=False, training=True, llm_client_factory=FixedModel,
                incremental_medicines=True, **case_paths,
            )
            workers = [session.runtime._workers[role][0] for role in CAPABILITIES]
            original = session._deliver

            def crash_before_medicine_reply(kind, source, capability, text,
                                            correlation_id=None, binary=None,
                                            shared_fields=None):
                if capability == 'retrieval.medicines':
                    workers[1].terminate()
                    workers[1].join()
                return original(kind, source, capability, text,
                                correlation_id, binary, shared_fields)

            try:
                with patch.object(session, '_deliver', side_effect=crash_before_medicine_reply):
                    with self.assertRaisesRegex(RuntimeError, '已退出'):
                        session.run_case(cases[0])
                self.assertIsNone(session._medicine_context)
                self.assertIsNone(session.runtime.trace_id)
                self.assertEqual(session.state, HospitalSessionState.FAILED)
                self.assertEqual(session.runtime.state, IpcRuntimeState.FAILED)
            finally:
                session.close()
            self.assertTrue(all(not worker.is_alive() for worker in workers))
            with ProcessHospitalSession(
                retrieval_mode='jaccard', verbose=False, training=True, llm_client_factory=FixedModel,
                incremental_medicines=True, **case_paths,
            ) as rebuilt:
                self.assertEqual(rebuilt.run_case(cases[1])['case_id'], cases[1].case_id)
                self.assertIsNone(rebuilt._medicine_context)

    def test_rejects_extra_labels_wrong_names_and_correlated_reply(self):
        context = {'trace_id': 77, 'evaluation_message_id': 5, 'case_id': 1,
                   'recommended_drugs': ['酮康唑'], 'actual_drugs': ['黄连素']}
        request = AgentPacket.create(
            trace_id=77, message_type=MessageType.REQUEST,
            source='ExpertAgent', correlation_id=5,
            payload={'capability': 'retrieval.medicines', 'text': '{}'},
        )
        correct = {'case_id': 1, 'medicine_introductions': {
            '酮康唑': {'说明': '内容'}, '黄连素': {'说明': '内容'},
        }}
        text = json.dumps(correct, ensure_ascii=False)
        response = AgentPacket.create(
            trace_id=77, message_type=MessageType.RESPONSE,
            source='RetrievalAgent', correlation_id=request.head.message_id,
            payload={'capability': 'retrieval.medicines', 'text': text},
        )
        self.assertEqual(json.loads(_restore_medicine_response(
            context, text, request, response)), {
                'case_id': 1, 'recommended_drugs': ['酮康唑'],
                'actual_drugs': ['黄连素'],
                'medicine_introductions': correct['medicine_introductions'],
            })
        bad_cases = (
            {**correct, 'actual_drugs': ['黄连素']},
            {'case_id': 2, 'medicine_introductions': correct['medicine_introductions']},
            {'case_id': 1, 'medicine_introductions': {'酮康唑': {'说明': '内容'}}},
        )
        for bad in bad_cases:
            bad_text = json.dumps(bad, ensure_ascii=False)
            bad_response = AgentPacket(response.head, {
                'capability': 'retrieval.medicines', 'text': bad_text,
            })
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                _restore_medicine_response(context, bad_text, request, bad_response)
        for bad_head in (
            replace(response.head, correlation_id=request.head.message_id + 1),
            replace(response.head, trace_id=78),
        ):
            with self.subTest(bad_head=bad_head), self.assertRaisesRegex(ValueError, '消息关联'):
                _restore_medicine_response(
                    context, text, request, AgentPacket(bad_head, response.payload))


if __name__ == '__main__':
    unittest.main()
