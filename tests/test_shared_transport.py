import json
import os
import sys
import tempfile
import time
import unittest
from multiprocessing import shared_memory
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import shared_transport  # noqa: E402
from shared_transport import attach_shared_memory  # noqa: E402
from cases import DIALMED_TRAIN_PATH, load_dialmed_cases  # noqa: E402
from hospital_process import HospitalSessionState, ProcessHospitalSession  # noqa: E402
from ipc_runtime import IpcRuntimeState, MultiprocessRuntime  # noqa: E402
from main import HospitalSession  # noqa: E402
from performance_baseline import FixedModel  # noqa: E402
from protocol import AgentPacket, MessageType  # noqa: E402
from tool.tracing import Trace  # noqa: E402


def short_echo(text):
    return str(len(text))


def plain_echo(text):
    return text


def crash_handler(_text):
    os._exit(31)


def slow_handler(text):
    time.sleep(0.3)
    return text


class CrashDoctor(FixedModel):
    def chat(self, messages):
        if messages[0]['content'].startswith(('你是一名专业的医生', '你是一名医生')):
            os._exit(32)
        return super().chat(messages)


class SharedTransportTests(unittest.TestCase):
    @staticmethod
    def _packet(runtime, text):
        return AgentPacket.create(
            trace_id=runtime.trace_id, message_type=MessageType.REQUEST,
            source='System', payload={'capability': 'demo.echo', 'text': text},
        )

    @staticmethod
    def _paths(root):
        return {
            'golden_path': root / 'golden.json',
            'reflection_path': root / 'reflections.json',
            'communication_path': root / 'communications.jsonl',
            'control_path': root / 'control.jsonl',
            'llm_path': root / 'llm.jsonl',
            'dataset_path': DIALMED_TRAIN_PATH,
        }

    def test_arena_reuses_bytes_validates_reference_and_unlinks(self):
        arena = shared_transport.SharedPayloadArena()
        try:
            data = '重复对话' * 500
            ref, _ns, written = arena.publish(data.encode('utf-8'), 77)
            other, repeat_ns, repeat_written = arena.publish(data.encode('utf-8'), 77)
            self.assertEqual(ref, other)
            self.assertEqual((written, repeat_ns, repeat_written),
                             (len(data.encode('utf-8')), 0, 0))
            self.assertEqual(arena.allocations, 1)
            self.assertEqual(shared_transport.read_reference(ref, 77)[0], data.encode('utf-8'))
            with self.assertRaisesRegex(ValueError, '病例不匹配'):
                shared_transport.read_reference(ref, 78)
            with self.assertRaisesRegex(ValueError, '内容校验'):
                shared_transport.read_reference({**ref, 'sha256': '0' * 64}, 77)
        finally:
            statistics = arena.release()
        self.assertEqual(statistics['allocations'], 1)
        self.assertEqual(arena.active_bytes, 0)
        with self.assertRaisesRegex(ValueError, '已失效'):
            shared_transport.read_reference(ref, 77)

    def test_generic_whole_packet_and_nested_field_paths(self):
        trace = Trace(verbose=False)
        with MultiprocessRuntime(trace) as runtime:
            runtime.register_process_agent('EchoAgent', 'demo.echo', plain_echo)
            runtime.start_case()
            trace.begin_case(runtime.trace_id)
            first = self._packet(runtime, '完整载荷' * 1200)
            whole = runtime.deliver_shared(first, min_bytes=128)
            self.assertEqual(whole.output, first.payload['text'])
            self.assertEqual(whole.packet.head.correlation_id, first.head.message_id)
            nested = json.dumps({'outer': [{'item': '嵌套数据' * 900}], 'tag': 'safe'},
                                ensure_ascii=False)
            second = self._packet(runtime, nested)
            field = runtime.deliver_shared(second, fields=('/outer/0/item',), min_bytes=128)
            self.assertEqual(json.loads(field.output), json.loads(nested))
            self.assertEqual(runtime.transport_frame_stats[-2]['mode'], 'shared')
            self.assertEqual(runtime.transport_frame_stats[-1]['mode'], 'shared')
            self.assertLess(runtime.transport_frame_stats[-1]['request_frame_bytes'],
                            runtime.transport_frame_stats[-1]['logical_request_bytes'])
            small = self._packet(runtime, '{"outer":[{"item":"短"}]}')
            self.assertEqual(runtime.deliver_shared(small, fields=('/outer/0/item',),
                                                    min_bytes=128).output, small.payload['text'])
            self.assertEqual(runtime.transport_frame_stats[-1]['mode'], 'pipe')
            names = runtime._shared_arena.active_names
            runtime.finish_case()
            self.assertEqual(runtime.shared_memory_active_bytes, 0)
            self.assertEqual(runtime.shared_memory_case_stats[-1]['allocations'], 2)
            for name in names:
                with self.assertRaises(FileNotFoundError):
                    attach_shared_memory(name)

    def test_corrupt_reference_returns_correlated_error_and_releases(self):
        trace = Trace(verbose=False)
        with MultiprocessRuntime(trace) as runtime:
            runtime.register_process_agent('EchoAgent', 'demo.echo', short_echo)
            runtime.start_case()
            trace.begin_case(runtime.trace_id)
            packet = self._packet(runtime, json.dumps({'payload': 'x' * 5000}))
            original = shared_transport.prepare_shared_request

            def corrupt(*args):
                wire, referenced, written, write_ns = original(*args)
                prefix = len(shared_transport._MARKER) + shared_transport._LENGTH.size
                length = shared_transport._LENGTH.unpack(
                    wire[len(shared_transport._MARKER):prefix])[0]
                meta = json.loads(wire[prefix:prefix + length])
                meta['fields'][0]['ref']['sha256'] = '0' * 64
                return shared_transport._envelope(meta, wire[prefix + length:]), referenced, written, write_ns

            with patch('ipc_runtime.prepare_shared_request', side_effect=corrupt):
                delivery = runtime.deliver_shared(packet, fields=('/payload',), min_bytes=128)
            self.assertEqual(delivery.packet.head.message_type, MessageType.ERROR.value)
            self.assertEqual(delivery.packet.head.correlation_id, packet.head.message_id)
            self.assertEqual(delivery.packet.head.trace_id, packet.head.trace_id)
            self.assertIn('内容校验失败', delivery.packet.payload['error'])
            self.assertEqual(runtime.state, IpcRuntimeState.CASE_ACTIVE)
            self.assertGreater(runtime.shared_memory_active_bytes, 0)
            runtime.finish_case()
            self.assertEqual(runtime.shared_memory_active_bytes, 0)

    def test_crash_during_shared_transfer_retires_worker_and_unlinks(self):
        trace = Trace(verbose=False)
        with MultiprocessRuntime(trace) as runtime:
            runtime.register_process_agent('EchoAgent', 'demo.echo', crash_handler)
            runtime.start_case()
            trace.begin_case(runtime.trace_id)
            packet = self._packet(runtime, 'x' * 100_000)
            worker = runtime._workers['EchoAgent'][0]
            response = runtime.deliver_shared(packet, min_bytes=128).packet
            self.assertEqual(response.head.message_type, MessageType.ERROR.value)
            self.assertEqual(response.head.correlation_id, packet.head.message_id)
            self.assertEqual(runtime.state, IpcRuntimeState.FAILED)
            self.assertEqual(runtime.shared_memory_active_bytes, 0)
            self.assertFalse(worker.is_alive())

    def test_timeout_during_shared_transfer_retires_worker_and_unlinks(self):
        trace = Trace(verbose=False)
        with MultiprocessRuntime(trace, timeout=0.1) as runtime:
            runtime.register_process_agent('EchoAgent', 'demo.echo', slow_handler)
            runtime.start_case()
            trace.begin_case(runtime.trace_id)
            packet = self._packet(runtime, 'x' * 100_000)
            response = runtime.deliver_shared(packet, min_bytes=128).packet
            self.assertEqual(response.head.message_type, MessageType.ERROR.value)
            self.assertIn('超时', response.payload['error'])
            self.assertEqual(runtime.state, IpcRuntimeState.FAILED)
            self.assertEqual(runtime.shared_memory_active_bytes, 0)

    def test_ten_cases_match_sync_and_pipe_and_keep_labels_private(self):
        cases = load_dialmed_cases(DIALMED_TRAIN_PATH, 10)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch('main.LLMClient', side_effect=FixedModel):
                sync = HospitalSession(retrieval_mode='jaccard', verbose=False, training=False,
                                       **self._paths(root / 'sync'))
                try:
                    expected = [sync.run_case(case) for case in cases]
                finally:
                    sync.close()
            with ProcessHospitalSession(
                retrieval_mode='jaccard', verbose=False, training=False, llm_client_factory=FixedModel,
                **self._paths(root / 'pipe'),
            ) as pipe:
                baseline = [pipe.run_case(case) for case in cases]
            with ProcessHospitalSession(
                retrieval_mode='jaccard', verbose=False, training=False, llm_client_factory=FixedModel,
                shared_dialog=True, **self._paths(root / 'shared'),
            ) as shared:
                outputs = []
                published = []
                original_publish = shared.runtime._shared_arena.publish

                def capture_publish(data, trace_id):
                    published.append(data)
                    return original_publish(data, trace_id)

                with patch.object(shared.runtime._shared_arena, 'publish',
                                  side_effect=capture_publish):
                    for case in cases:
                        outputs.append(shared.run_case(case))
                        self.assertEqual(shared.runtime.shared_memory_active_bytes, 0)
                self.assertEqual(set(published),
                                 {case.patient_input.encode('utf-8') for case in cases
                                  if len(case.patient_input.encode('utf-8')) >= 256})
                self.assertEqual(outputs, baseline)
                self.assertEqual(outputs, expected)
                rows = [row for row in shared.runtime.transport_frame_stats
                        if row['mode'] == 'shared']
                eligible = sum(len(case.patient_input.encode("utf-8")) >= 256 for case in cases)
                self.assertEqual(len(rows), eligible * 3)
                self.assertEqual(sum(row['shared_new_bytes'] > 0 for row in rows), eligible)
                self.assertEqual(len(shared.runtime.shared_memory_case_stats), 10)
                self.assertEqual(sum(row['allocations'] for row in
                                     shared.runtime.shared_memory_case_stats), eligible)
                doctor_inputs = [packet for packet in shared.trace.packets
                                 if packet.payload.get('capability') == 'doctor.recommend'
                                 and packet.head.source == 'RetrievalAgent']
                self.assertEqual(len(doctor_inputs), 10)
                self.assertTrue(all(not ({'actual_drugs', 'drug_labels'}
                                      & set(json.loads(packet.payload['text'])))
                                    for packet in doctor_inputs))
                for case in cases:
                    packet = next(packet for packet in doctor_inputs
                                  if json.loads(packet.payload['text'])['case_id'] == case.case_id)
                    self.assertEqual(json.loads(packet.payload['text'])['patient_dialog'],
                                     case.patient_input)

    def test_hospital_crash_releases_shared_case_and_new_session_runs(self):
        cases = load_dialmed_cases(DIALMED_TRAIN_PATH, 2)
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._paths(Path(temporary))
            session = ProcessHospitalSession(
                retrieval_mode='jaccard', verbose=False, training=False, llm_client_factory=CrashDoctor,
                shared_dialog=True, **paths,
            )
            try:
                with self.assertRaisesRegex(RuntimeError, '连接中断|已退出'):
                    session.run_case(cases[0])
                self.assertEqual(session.state, HospitalSessionState.FAILED)
                self.assertEqual(session.runtime.shared_memory_active_bytes, 0)
                self.assertEqual(session.runtime.shared_memory_case_stats[-1]['allocations'], 1)
                self.assertEqual(session.runtime.shared_memory_case_stats[-1]['cleanup_errors'], [])
            finally:
                session.close()
            with ProcessHospitalSession(
                retrieval_mode='jaccard', verbose=False, training=False, llm_client_factory=FixedModel,
                shared_dialog=True, **paths,
            ) as rebuilt:
                self.assertEqual(rebuilt.run_case(cases[1])['case_id'], cases[1].case_id)


if __name__ == '__main__':
    unittest.main()
