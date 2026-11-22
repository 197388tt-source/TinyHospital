import os
import multiprocessing as mp
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ipc_runtime import MultiprocessRuntime  # noqa: E402
from protocol import AgentPacket, MessageType  # noqa: E402
from tool.tracing import Trace  # noqa: E402


def deterministic_echo(text: str) -> str:
    return f"收到:{text}"


def failing_handler(_text: str) -> str:
    raise ValueError("模拟处理失败")


def slow_handler(text: str) -> str:
    time.sleep(0.3)
    return text


def exiting_handler(_text: str) -> str:
    os._exit(7)


def stalled_reader(_requests, replies, _name, _capability, _handler, _timeout):
    replies.send_bytes(b"READY")
    time.sleep(2)


def never_ready(_requests, _replies, _name, _capability, _handler, _timeout):
    time.sleep(2)


class SenderAgent:
    name = "SenderAgent"
    runtime = None

    def make_packet(self, message_type, *, payload):
        return AgentPacket.create(trace_id=0, message_type=message_type,
                                  source=self.name, payload=payload)


class IpcRuntimeTests(unittest.TestCase):
    def _request(self, runtime: MultiprocessRuntime, text: str,
                 *, trace_id: int | None = None) -> AgentPacket:
        return AgentPacket.create(
            trace_id=runtime.trace_id if trace_id is None else trace_id,
            message_type=MessageType.REQUEST, source="System",
            payload={"capability": "demo.echo", "text": text},
        )

    def test_roundtrip_ids_trace_and_real_process(self):
        trace = Trace(verbose=False)
        with MultiprocessRuntime(trace) as runtime:
            runtime.register_process_agent("EchoAgent", "demo.echo", deterministic_echo)
            self.assertEqual(runtime.capabilities.resolve("demo.echo"), "EchoAgent")
            runtime.start_case()
            trace.begin_case(runtime.trace_id)
            replies = []
            for value in ("你好", "第二条"):
                request = self._request(runtime, value)
                delivery = runtime.deliver(request)
                response = delivery.packet
                self.assertEqual(response.head.message_type, MessageType.RESPONSE.value)
                self.assertEqual(response.head.trace_id, request.head.trace_id)
                self.assertEqual(response.head.correlation_id, request.head.message_id)
                self.assertNotEqual(response.head.message_id, request.head.message_id)
                self.assertEqual(response.head.source, "EchoAgent")
                self.assertEqual(delivery.output, f"收到:{value}")
                self.assertNotEqual(runtime._workers["EchoAgent"][0].pid, os.getpid())
                replies.append(response)
            ids = [packet.head.message_id for packet in trace.packets
                   if packet.head.trace_id == runtime.trace_id]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual([packet.head.message_id for packet in replies], ids[1::2])
            process = runtime._workers["EchoAgent"][0]
        self.assertFalse(process.is_alive())

    def test_handler_error_is_correlated_and_process_survives(self):
        trace = Trace(verbose=False)
        with MultiprocessRuntime(trace) as runtime:
            runtime.register_process_agent("EchoAgent", "demo.echo", failing_handler)
            runtime.start_case()
            trace.begin_case(runtime.trace_id)
            request = self._request(runtime, "输入")
            reply = runtime.deliver(request).packet
            self.assertEqual(reply.head.message_type, MessageType.ERROR.value)
            self.assertEqual(reply.head.correlation_id, request.head.message_id)
            self.assertEqual(reply.head.trace_id, request.head.trace_id)
            self.assertIn("模拟处理失败", reply.payload["error"])
            self.assertTrue(runtime._workers["EchoAgent"][0].is_alive())

    def test_response_is_forwarded_to_child_and_acknowledged(self):
        trace = Trace(verbose=False)
        with MultiprocessRuntime(trace) as runtime:
            runtime.register_process_agent("EchoAgent", "demo.echo", deterministic_echo)
            runtime.register_agent(SenderAgent(), {"sender.noop": deterministic_echo})
            runtime.start_case()
            trace.begin_case(runtime.trace_id)
            sent = AgentPacket.create(
                trace_id=runtime.trace_id, message_type=MessageType.RESPONSE,
                source="SenderAgent", correlation_id=1,
                payload={"capability": "demo.echo", "text": "专家结果"},
            )
            delivery = runtime.deliver(sent)
            self.assertEqual(delivery.packet.head.message_type, MessageType.RESPONSE.value)
            self.assertEqual(delivery.packet.head.correlation_id, sent.head.message_id)
            self.assertEqual(delivery.packet.head.trace_id, sent.head.trace_id)
            self.assertEqual(delivery.output, "收到:专家结果")
            self.assertEqual(trace.messages[-1], ("SenderAgent", "EchoAgent", "专家结果"))

    def test_ready_timeout_cleans_up_without_registration(self):
        trace = Trace(verbose=False)
        before = {process.pid for process in mp.active_children()}
        with patch("ipc_runtime._agent_worker", never_ready):
            with MultiprocessRuntime(trace, startup_timeout=0.05) as runtime:
                with self.assertRaisesRegex(RuntimeError, "就绪超时"):
                    runtime.register_process_agent("EchoAgent", "demo.echo", deterministic_echo)
                self.assertIsNone(runtime.capabilities.resolve("demo.echo"))
                self.assertNotIn("EchoAgent", runtime.agents)
        self.assertEqual({process.pid for process in mp.active_children()}, before)

    def test_blocked_send_has_deadline_and_retires_child(self):
        trace = Trace(verbose=False)
        with patch("ipc_runtime._agent_worker", stalled_reader):
            with MultiprocessRuntime(trace, timeout=0.1) as runtime:
                runtime.register_process_agent("EchoAgent", "demo.echo", deterministic_echo)
                runtime.start_case()
                trace.begin_case(runtime.trace_id)
                process = runtime._workers["EchoAgent"][0]
                request = self._request(runtime, "大" * 400_000)
                started = time.monotonic()
                error = runtime.deliver(request).packet
                self.assertLess(time.monotonic() - started, 1.0)
                self.assertEqual(error.head.message_type, MessageType.ERROR.value)
                self.assertEqual(error.head.correlation_id, request.head.message_id)
                self.assertIn("超时", error.payload["error"])
                self.assertFalse(process.is_alive())

    def test_timeout_retires_child(self):
        trace = Trace(verbose=False)
        with MultiprocessRuntime(trace, timeout=0.2) as runtime:
            runtime.register_process_agent("EchoAgent", "demo.echo", slow_handler)
            runtime.start_case()
            trace.begin_case(runtime.trace_id)
            process = runtime._workers["EchoAgent"][0]
            request = self._request(runtime, "输入")
            error = runtime.deliver(request).packet
            self.assertEqual(error.head.message_type, MessageType.ERROR.value)
            self.assertEqual(error.head.correlation_id, request.head.message_id)
            self.assertIn("超时", error.payload["error"])
            self.assertFalse(process.is_alive())
            self.assertNotIn("EchoAgent", runtime._workers)

    def test_unexpected_exit_returns_error_and_cleans_up(self):
        trace = Trace(verbose=False)
        with MultiprocessRuntime(trace) as runtime:
            runtime.register_process_agent("EchoAgent", "demo.echo", exiting_handler)
            runtime.start_case()
            trace.begin_case(runtime.trace_id)
            process = runtime._workers["EchoAgent"][0]
            request = self._request(runtime, "输入")
            error = runtime.deliver(request).packet
            self.assertEqual(error.head.message_type, MessageType.ERROR.value)
            self.assertEqual(error.head.correlation_id, request.head.message_id)
            self.assertIn("连接中断", error.payload["error"])
            self.assertFalse(process.is_alive())

    def test_wrong_trace_is_rejected_before_ipc(self):
        trace = Trace(verbose=False)
        with MultiprocessRuntime(trace) as runtime:
            runtime.register_process_agent("EchoAgent", "demo.echo", deterministic_echo)
            runtime.start_case()
            trace.begin_case(runtime.trace_id)
            request = self._request(runtime, "输入", trace_id=runtime.trace_id + 100)
            error = runtime.deliver(request).packet
            self.assertIn("trace_id", error.payload["error"])
            self.assertEqual(error.head.correlation_id, request.head.message_id)


if __name__ == "__main__":
    unittest.main()
