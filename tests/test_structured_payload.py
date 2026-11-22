"""Native TH-ACP data payload compatibility and shared field delivery."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cases import DIALMED_TRAIN_PATH, load_dialmed_cases  # noqa: E402
from hospital_process import ProcessHospitalSession  # noqa: E402
from ipc_runtime import MultiprocessRuntime  # noqa: E402
from performance_baseline import FixedModel  # noqa: E402
from protocol import AgentPacket, MessageType, ProtocolParser, Runtime  # noqa: E402
from tool.tracing import Trace  # noqa: E402


def echo(value):
    return value


class StructuredPayloadTests(unittest.TestCase):
    def test_data_text_and_parameters_are_distinct_valid_shapes(self):
        parser = ProtocolParser()
        data = AgentPacket.create(trace_id=1, message_type=MessageType.REQUEST,
                                  source="System", payload={"capability": "demo.echo",
                                                            "data": {"case_id": 1, "items": ["药"]}})
        self.assertEqual(parser.parse(data.to_json()), data)
        self.assertNotIn('\\"case_id\\"', data.to_json())
        for body in ({"capability": "demo.echo", "data": [1]},
                     {"capability": "demo.echo", "data": {"bad": float("nan")}},
                     {"capability": "demo.echo", "data": {}, "text": "{}"}):
            packet = AgentPacket.create(trace_id=1, message_type=MessageType.REQUEST,
                                        source="System", payload=body)
            self.assertEqual(parser.parse(packet.to_json()).head.message_type, "ERROR")
        legacy = AgentPacket.create(trace_id=1, message_type=MessageType.REQUEST,
                                    source="System", payload={"capability": "demo.echo",
                                                              "text": "普通文本"})
        self.assertEqual(parser.parse(legacy.to_json()), legacy)
        runtime = Runtime(Trace(verbose=False))
        runtime.start_case()
        malformed = AgentPacket.create(
            trace_id=runtime.trace_id, message_type=MessageType.REQUEST,
            source="System", payload={"capability": "demo.echo", "data": {1: "bad", "ok": 1}})
        error = runtime.deliver(malformed).packet
        self.assertEqual(error.head.message_type, "ERROR")
        self.assertEqual(error.head.correlation_id, malformed.head.message_id)

    def test_shared_data_field_restores_and_cleans(self):
        trace = Trace(verbose=False)
        with MultiprocessRuntime(trace) as runtime:
            runtime.register_process_agent("EchoAgent", "demo.echo", echo)
            runtime.start_case()
            trace.begin_case(runtime.trace_id)
            body = {"case_id": 1, "patient_dialog": "对话" * 200, "count": 1}
            packet = AgentPacket.create(trace_id=runtime.trace_id,
                                        message_type=MessageType.REQUEST, source="System",
                                        payload={"capability": "demo.echo", "data": body})
            result = runtime.deliver_shared(packet, fields=("/patient_dialog",), min_bytes=128)
            self.assertEqual(json.loads(result.output), body)
            self.assertEqual(packet.payload["data"], body)
            self.assertEqual(runtime.transport_frame_stats[-1]["mode"], "shared")
            runtime.finish_case()
            self.assertEqual(runtime.shared_memory_active_bytes, 0)
            self.assertEqual(runtime.shared_memory_case_stats[-1]["cleanup_errors"], [])

    def test_full_features_one_case_matches_legacy_and_keeps_labels_private(self):
        case = load_dialmed_cases(DIALMED_TRAIN_PATH, 1)[0]
        options = {"shared_dialog": True, "shared_medicines": True,
                   "incremental_history": True, "incremental_medicines": True}
        with tempfile.TemporaryDirectory() as directory:
            outcomes = []
            prompts = []
            for structured in (False, True):
                root = Path(directory) / str(structured)
                with ProcessHospitalSession(
                    retrieval_mode='jaccard', verbose=False, training=False, llm_client_factory=FixedModel,
                    golden_path=root / "golden.json", reflection_path=root / "reflection.json",
                    communication_path=root / "communication.jsonl",
                    control_path=root / "control.jsonl", llm_path=root / "llm.jsonl",
                    structured_data=structured, **options,
                ) as session:
                    outcomes.append(session.run_case(case))
                    doctor = next(packet for packet in session.trace.packets
                                  if packet.payload.get("capability") == "doctor.recommend"
                                  and packet.head.source == "RetrievalAgent")
                    body = doctor.payload.get("data") if structured else json.loads(doctor.payload["text"])
                    self.assertNotIn("actual_drugs", body)
                    self.assertNotIn("drug_labels", body)
                    self.assertEqual(session.runtime.shared_memory_active_bytes, 0)
                    self.assertEqual(session.runtime.shared_memory_case_stats[-1]["cleanup_errors"], [])
                    prompts.append((session.runtime.control_agent("DoctorAgent", "snapshot")["prompts"],
                                    session.runtime.control_agent("ExpertAgent", "snapshot")["prompts"]))
            self.assertEqual(outcomes[0], outcomes[1])
            self.assertEqual(prompts[0], prompts[1])

    def test_data_medicine_bad_reference_recovers_without_leak(self):
        cases = load_dialmed_cases(DIALMED_TRAIN_PATH, 2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with ProcessHospitalSession(
                retrieval_mode='jaccard', verbose=False, training=True, llm_client_factory=FixedModel,
                golden_path=root / "golden.json", reflection_path=root / "reflection.json",
                communication_path=root / "communication.jsonl",
                control_path=root / "control.jsonl", llm_path=root / "llm.jsonl",
                shared_dialog=True, shared_medicines=True,
                incremental_history=True, incremental_medicines=True,
                structured_data=True,
            ) as session:
                original = session._deliver

                def corrupt(kind, source, capability, text, correlation_id=None,
                            binary=None, shared_fields=None):
                    if capability == "expert.review":
                        body = json.loads(text)
                        body["shared_medicine_ref"]["sha256"] = "0" * 64
                        text = json.dumps(body, ensure_ascii=False)
                    return original(kind, source, capability, text,
                                    correlation_id, binary, shared_fields)

                with patch.object(session, "_deliver", side_effect=corrupt):
                    with self.assertRaisesRegex(RuntimeError, "共享药物说明引用"):
                        session.run_case(cases[0])
                self.assertEqual(session.runtime.shared_memory_active_bytes, 0)
                self.assertEqual(session.run_case(cases[1])["case_id"], cases[1].case_id)
                self.assertEqual(session.runtime.shared_memory_active_bytes, 0)


if __name__ == "__main__":
    unittest.main()
