import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import Agent, Trace  # noqa: E402
from protocol import AgentPacket, LocalIds, MessageType, ProtocolParser, RETRIEVAL_CAPABILITY, Runtime  # noqa: E402


class ProtocolTests(unittest.TestCase):
    def test_structured_retrieval_request(self):
        trace = Trace(verbose=False)
        runtime = Runtime(trace)
        target = Agent("RetrievalAgent", trace)
        runtime.register_agent(target, {RETRIEVAL_CAPABILITY: lambda parameters: parameters})
        runtime.start_case()
        trace.begin_case(runtime.trace_id)
        parameters = {
            "candidate_diseases": ["流行性感冒", "普通感冒"],
            "knowledge_type": "symptoms",
            "include_history": True,
        }
        packet = AgentPacket.create(
            trace_id=runtime.trace_id, message_type=MessageType.REQUEST,
            source="System",
            payload={"capability": RETRIEVAL_CAPABILITY, "parameters": parameters},
        )
        self.assertEqual(runtime.parser.parse(packet.to_json()), packet)
        self.assertEqual(runtime.deliver(packet).output, parameters)
        self.assertEqual(json.loads(trace.messages[-1][2]), parameters)
        separate = json.loads(packet.to_json())
        separate["payload"]["parameters"]["memory_types"] = ["case", "experience"]
        self.assertEqual(runtime.parser.parse(separate).head.message_type, MessageType.REQUEST.value)
        for invalid_payload in (
            {"capability": RETRIEVAL_CAPABILITY, "text": "旧版检索请求"},
            {"capability": RETRIEVAL_CAPABILITY, "parameters": parameters | {"extra": 1}},
            {"capability": RETRIEVAL_CAPABILITY, "parameters": parameters | {"include_history": 1}},
            {"capability": RETRIEVAL_CAPABILITY, "parameters": parameters | {"candidate_diseases": []}},
            {"capability": RETRIEVAL_CAPABILITY, "parameters": parameters | {"memory_types": ["pending"]}},
        ):
            invalid = json.loads(packet.to_json())
            invalid["payload"] = invalid_payload
            self.assertEqual(
                runtime.parser.parse(invalid).head.message_type,
                MessageType.ERROR.value,
            )

    def test_message_specific_payload_and_errors(self):
        parser = ProtocolParser()
        malformed = parser.parse('{"message_type": "REQUEST"}')
        self.assertEqual(malformed.head.message_type, MessageType.ERROR.value)
        self.assertIn("head 和 payload", malformed.payload["error"])

        trace = Trace(verbose=False)
        runtime = Runtime(trace)
        runtime.start_case()
        samples = [
            AgentPacket.create(
                trace_id=0, message_type=MessageType.HELLO,
                source="Agent", payload={"capabilities": ["agent.echo"]},
            ),
            AgentPacket.create(
                trace_id=0, message_type=MessageType.HELLO_ACK,
                source="Runtime", correlation_id=1,
                payload={"status": "ok", "capabilities": ["agent.echo"]},
            ),
            AgentPacket.create(
                trace_id=runtime.trace_id, message_type=MessageType.REQUEST,
                source="Agent", payload={"capability": "agent.echo", "text": "你好"},
            ),
            AgentPacket.create(
                trace_id=runtime.trace_id, message_type=MessageType.RESPONSE,
                source="Agent", correlation_id=1,
                payload={"capability": "agent.echo", "text": "收到"},
            ),
            AgentPacket.create(
                trace_id=runtime.trace_id, message_type=MessageType.ERROR,
                source="Runtime", correlation_id=1, payload={"error": "错误"},
            ),
        ]
        for packet in samples:
            encoded = json.loads(packet.to_json())
            self.assertEqual(set(encoded), {"head", "payload"})
            self.assertNotIn("target", packet.to_json())
            self.assertNotIn("action", packet.to_json())
            self.assertEqual(parser.parse(packet.to_json()), packet)
            invalid = encoded | {"payload": encoded["payload"] | {"unused": None}}
            error = parser.parse(invalid)
            self.assertEqual(error.head.message_type, MessageType.ERROR.value)
            self.assertEqual(error.head.correlation_id, packet.head.message_id)
            self.assertIn("payload", error.payload["error"])

        request = samples[2]
        missing = json.loads(request.to_json())
        del missing["payload"]["capability"]
        self.assertIn("capability", parser.parse(missing).payload["error"])
        old_shape = json.loads(request.to_json()) | {"target": "Agent"}
        self.assertIn("head 和 payload", parser.parse(old_shape).payload["error"])
        for name, invalid_value in (
            ("message_id", "1"), ("message_id", True), ("message_id", 0),
            ("trace_id", "1"), ("trace_id", -1),
            ("correlation_id", "1"), ("correlation_id", 0),
        ):
            invalid = json.loads(request.to_json())
            invalid["head"][name] = invalid_value
            self.assertIn(name, parser.parse(invalid).payload["error"])
        wrong_trace = json.loads(request.to_json())
        wrong_trace["head"]["trace_id"] = 0
        self.assertIn("病例 trace_id", parser.parse(wrong_trace).payload["error"])

    def test_local_ids_increment_per_trace(self):
        ids = LocalIds()
        self.assertEqual((ids.new_trace(), ids.new_trace()), (1, 2))
        self.assertEqual([ids.next_message(0) for _ in range(2)], [1, 2])
        self.assertEqual([ids.next_message(1) for _ in range(3)], [1, 2, 3])
        self.assertEqual([ids.next_message(2) for _ in range(2)], [1, 2])
        first = Runtime(Trace(verbose=False))
        second = Runtime(Trace(verbose=False))
        first_trace = first.start_case()
        self.assertEqual(second.start_case(), first_trace + 1)

    def test_capability_routing_and_handler_mapping(self):
        trace = Trace(verbose=False)
        runtime = Runtime(trace)
        target = Agent("TargetAgent", trace)
        runtime.register_agent(target, {"target.echo": lambda text: text + "！"})
        self.assertEqual(runtime.capabilities.resolve("target.echo"), "TargetAgent")
        self.assertIn("TargetAgent", runtime.agents)
        self.assertEqual(len(trace.packets), 2)
        registration = trace.packets[0]
        acknowledgement = next(
            packet for packet in trace.packets
            if packet.head.message_type == MessageType.HELLO_ACK.value
        )
        self.assertEqual(registration.payload["capabilities"], ["target.echo"])
        self.assertEqual(registration.head.trace_id, 0)
        self.assertEqual(acknowledgement.head.trace_id, 0)
        self.assertEqual(acknowledgement.head.correlation_id, registration.head.message_id)
        self.assertEqual(acknowledgement.payload["capabilities"], ["target.echo"])
        runtime.start_case()
        trace.begin_case(runtime.trace_id)
        request = AgentPacket.create(
            trace_id=runtime.trace_id, message_type=MessageType.REQUEST,
            source="System", payload={"capability": "target.echo", "text": "你好"},
        )
        self.assertEqual(runtime.deliver(request).output, "你好！")
        self.assertEqual(trace.messages[-1][:2], ("System", "TargetAgent"))
        unknown = AgentPacket.create(
            trace_id=runtime.trace_id, message_type=MessageType.REQUEST,
            source="System", payload={"capability": "target.unknown", "text": "你好"},
        )
        error = runtime.deliver(unknown).packet
        self.assertEqual(error.head.message_type, MessageType.ERROR.value)
        self.assertIn("capability 未注册", error.payload["error"])
        self.assertEqual(error.head.correlation_id, unknown.head.message_id)
        end = AgentPacket.create(
            trace_id=runtime.trace_id, message_type=MessageType.RESPONSE,
            source="TargetAgent", correlation_id=request.head.message_id,
            payload={"capability": "case.complete", "text": "完成"},
        )
        runtime.deliver(end)
        self.assertEqual(trace.messages[-1][:2], ("TargetAgent", "End"))

    def test_hello_rejects_duplicate_capability_without_partial_registration(self):
        trace = Trace(verbose=False)
        runtime = Runtime(trace)
        first = Agent("FirstAgent", trace)
        runtime.register_agent(first, {"shared.echo": lambda text: text})
        second = Agent("SecondAgent", trace)
        with self.assertRaisesRegex(RuntimeError, "已注册"):
            runtime.register_agent(second, {
                "second.extra": lambda text: text,
                "shared.echo": lambda text: text,
            })
        self.assertNotIn("SecondAgent", runtime.agents)
        self.assertIsNone(runtime.capabilities.resolve("second.extra"))
        self.assertIsNone(runtime.actions.resolve("second.extra"))
        self.assertIsNone(second.runtime)


if __name__ == "__main__":
    unittest.main()
