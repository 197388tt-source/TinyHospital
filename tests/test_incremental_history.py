"""Stage 8: only the retrieval.history reply changes on the IPC wire."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cases import DIALMED_TRAIN_PATH, load_dialmed_cases  # noqa: E402
from hospital_process import (CAPABILITIES, HospitalSessionState,
                              ProcessHospitalSession, _restore_history_response)  # noqa: E402
from ipc_runtime import IpcRuntimeState  # noqa: E402
from performance_baseline import FixedModel  # noqa: E402
from protocol import Delivery  # noqa: E402


def paths(root, dataset=DIALMED_TRAIN_PATH):
    return {
        'golden_path': root / 'golden.json',
        'reflection_path': root / 'reflection.json',
        'communication_path': root / 'communication.jsonl',
        'control_path': root / 'control.jsonl',
        'llm_path': root / 'llm.jsonl',
        'dataset_path': dataset,
    }


def packet_texts(session, capability, source):
    return [packet.payload['text'] for packet in session.trace.packets
            if packet.payload.get('capability') == capability
            and packet.head.source == source]


class IncrementalHistoryTests(unittest.TestCase):
    def test_ten_cases_preserve_doctor_input_and_outputs(self):
        cases = load_dialmed_cases(DIALMED_TRAIN_PATH, 10)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with ProcessHospitalSession(
                retrieval_mode='jaccard', verbose=False, training=False, llm_client_factory=FixedModel,
                **paths(root / 'full'),
            ) as full:
                expected = [full.run_case(case) for case in cases]
                full_doctor = packet_texts(full, 'doctor.recommend', 'RetrievalAgent')
                full_replies = packet_texts(full, 'retrieval.history', 'RetrievalAgent')
            with ProcessHospitalSession(
                retrieval_mode='jaccard', verbose=False, training=False, llm_client_factory=FixedModel,
                incremental_history=True, **paths(root / 'delta'),
            ) as delta:
                workers = [delta.runtime._workers[role][0] for role in CAPABILITIES]
                actual = [delta.run_case(case) for case in cases]
                self.assertEqual(actual, expected)
                self.assertEqual(packet_texts(delta, 'doctor.recommend', 'RetrievalAgent'),
                                 full_doctor)
                self.assertEqual(len(delta.history_delta_stats), 10)
                self.assertIsNone(delta._history_context)
                self.assertEqual(delta.state, HospitalSessionState.READY)
                replies = packet_texts(delta, 'retrieval.history', 'RetrievalAgent')
                self.assertEqual(len(replies), 10)
                for full_reply, delta_reply, doctor_input in zip(
                        full_replies, replies, full_doctor):
                    self.assertEqual(set(json.loads(delta_reply)), {'case_id', 'history'})
                    self.assertLess(len(delta_reply.encode('utf-8')),
                                    len(full_reply.encode('utf-8')))
                    self.assertEqual(json.loads(doctor_input), json.loads(full_reply))
                    self.assertNotIn('actual_drugs', json.loads(doctor_input))
                    self.assertNotIn('drug_labels', json.loads(doctor_input))
                doctor_prompts = delta.runtime.control_agent('DoctorAgent', 'snapshot')['prompts']
                self.assertEqual(len(doctor_prompts), 10)
                self.assertTrue(all('actual_drugs' not in prompt[1]['content']
                                    and 'drug_labels' not in prompt[1]['content']
                                    for prompt in doctor_prompts))
                self.assertTrue(all(worker.is_alive() for worker in workers))
            self.assertTrue(all(not worker.is_alive() for worker in workers))

    def test_empty_and_nonempty_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / 'cases.jsonl'
            dialogs = ('患者：头皮屑伴瘙痒三天', '患者：头皮屑伴瘙痒四天')
            dataset.write_text(''.join(json.dumps({
                'dialog': [dialog], 'label': ['酮康唑'],
            }, ensure_ascii=False) + '\n' for dialog in dialogs), encoding='utf-8')
            case_paths = paths(root, dataset)
            case_paths['golden_path'].write_text(json.dumps([{
                'case_id': 1, 'experience': '头皮屑伴瘙痒时先核对药物适应症。',
            }], ensure_ascii=False), encoding='utf-8')
            cases = load_dialmed_cases(dataset, 2)
            with ProcessHospitalSession(
                retrieval_mode='jaccard', verbose=False, training=False, llm_client_factory=FixedModel,
                incremental_history=True, **case_paths,
            ) as session:
                for case in cases:
                    session.run_case(case)
                replies = [json.loads(text) for text in packet_texts(
                    session, 'retrieval.history', 'RetrievalAgent')]
                self.assertEqual(replies[0]['history'], {
                    'golden_experiences': [], 'reflections': [],
                })
                self.assertEqual(replies[1]['history']['golden_experiences'],
                                 ['头皮屑伴瘙痒时先核对药物适应症。'])
                self.assertIsNone(session._history_context)

    def test_shared_dialog_and_incremental_reply_are_independent(self):
        case = load_dialmed_cases(DIALMED_TRAIN_PATH, 1)[0]
        with tempfile.TemporaryDirectory() as temporary:
            with ProcessHospitalSession(
                retrieval_mode='jaccard', verbose=False, training=False, llm_client_factory=FixedModel,
                incremental_history=True, shared_dialog=True,
                **paths(Path(temporary)),
            ) as session:
                self.assertEqual(session.run_case(case)['case_id'], case.case_id)
                self.assertIsNone(session._history_context)
                self.assertEqual(session.runtime.shared_memory_active_bytes, 0)
                reply = packet_texts(session, 'retrieval.history', 'RetrievalAgent')[0]
                self.assertEqual(set(json.loads(reply)), {'case_id', 'history'})
                self.assertTrue(any(row['mode'] == 'shared' for row in
                                    session.runtime.transport_frame_stats))

    def test_bad_delta_aborts_and_next_case_runs(self):
        cases = load_dialmed_cases(DIALMED_TRAIN_PATH, 2)
        with tempfile.TemporaryDirectory() as temporary:
            with ProcessHospitalSession(
                retrieval_mode='jaccard', verbose=False, training=False, llm_client_factory=FixedModel,
                incremental_history=True, **paths(Path(temporary)),
            ) as session:
                original = session.runtime.deliver
                injected = False

                def corrupt(packet):
                    nonlocal injected
                    delivery = original(packet)
                    if packet.payload.get('capability') == 'retrieval.history' and not injected:
                        injected = True
                        return Delivery(delivery.packet, json.dumps({
                            'case_id': 9999, 'history': {
                                'golden_experiences': [], 'reflections': [],
                            },
                        }))
                    return delivery

                with patch.object(session.runtime, 'deliver', side_effect=corrupt):
                    with self.assertRaisesRegex(ValueError, '病例编号不匹配'):
                        session.run_case(cases[0])
                self.assertIsNone(session._history_context)
                self.assertIsNone(session.runtime.trace_id)
                self.assertEqual(session.state, HospitalSessionState.READY)
                self.assertEqual(session.runtime.state, IpcRuntimeState.READY)
                for role in CAPABILITIES:
                    snapshot = session.runtime.control_agent(role, 'snapshot')
                    self.assertEqual((snapshot['state'], snapshot['phase'], snapshot['trace_id']),
                                     ('IDLE', 'IDLE', None))
                self.assertEqual(session.run_case(cases[1])['case_id'], cases[1].case_id)
                self.assertIsNone(session._history_context)

    def test_delta_rejects_extra_fields_and_bad_history(self):
        context = {'case_id': 1, 'dataset_path': 'data/train.txt',
                   'patient_dialog': '["患者：腹痛"]', 'count': 1}
        for bad in (
            {'case_id': 1, 'history': {'golden_experiences': [], 'reflections': []},
             'actual_drugs': ['秘密标签']},
            {'case_id': 1, 'history': {'golden_experiences': [''], 'reflections': []}},
            {'case_id': 1, 'history': {'golden_experiences': []}},
            {'case_id': 2, 'history': {'golden_experiences': [], 'reflections': []}},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                _restore_history_response(context, json.dumps(bad, ensure_ascii=False))


if __name__ == '__main__':
    unittest.main()
