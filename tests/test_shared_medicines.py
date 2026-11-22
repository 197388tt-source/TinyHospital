"""Retrieval-owned medicine shared memory reaches Expert and is reclaimed."""

import json
import sys
import tempfile
import unittest
from multiprocessing import shared_memory
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared_transport import attach_shared_memory  # noqa: E402
from cases import DIALMED_TRAIN_PATH, load_dialmed_cases  # noqa: E402
from hospital_process import ProcessHospitalSession  # noqa: E402
from ipc_runtime import IpcRuntimeState  # noqa: E402
from ipc_runtime import MultiprocessRuntime  # noqa: E402
from performance_baseline import FixedModel  # noqa: E402
from shared_transport import read_reference  # noqa: E402
from tool.tracing import Trace  # noqa: E402


def paths(root):
    return {"golden_path": root / "golden.json",
            "reflection_path": root / "reflection.json",
            "communication_path": root / "communication.jsonl",
            "control_path": root / "control.jsonl",
            "llm_path": root / "llm.jsonl"}


def session(root, **options):
    return ProcessHospitalSession(
        retrieval_mode='jaccard', verbose=False, training=True, llm_client_factory=FixedModel,
        shared_dialog=True, incremental_history=True,
        incremental_medicines=True, **paths(root), **options)


class SharedMedicinesTests(unittest.TestCase):
    def test_shared_medicines_alone_works_with_full_retrieval_reply(self):
        case = load_dialmed_cases(DIALMED_TRAIN_PATH, 1)[0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with ProcessHospitalSession(
                retrieval_mode='jaccard', verbose=False, training=True, llm_client_factory=FixedModel,
                shared_medicines=True, **paths(root),
            ) as shared:
                self.assertEqual(shared.run_case(case)["case_id"], case.case_id)
                reply = next(packet for packet in shared.trace.packets
                             if packet.payload.get("capability") == "retrieval.medicines"
                             and packet.head.source == "RetrievalAgent")
                self.assertEqual(set(json.loads(reply.payload["text"])),
                                 {"case_id", "recommended_drugs", "actual_drugs",
                                  "medicine_introductions", "shared_medicine_ref"})
                self.assertEqual(shared.runtime.shared_memory_active_bytes, 0)

    def test_pre_reply_failure_reclaims_reserved_name(self):
        trace = Trace(verbose=False)
        with MultiprocessRuntime(trace, shared_medicines=True) as runtime:
            runtime.start_case()
            name = runtime.reserve_medicine_segment()
            segment = shared_memory.SharedMemory(name=name, create=True, size=16)
            segment.close()
            runtime.fail("simulated failure before Retrieval sent its reference")
            self.assertEqual(runtime.state, IpcRuntimeState.FAILED)
            with self.assertRaises(FileNotFoundError):
                attach_shared_memory(name)

    def test_expert_restores_exact_original_input_and_releases_segment(self):
        case = load_dialmed_cases(DIALMED_TRAIN_PATH, 1)[0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with session(root / "a") as plain:
                expected = plain.run_case(case)
                original = next(packet.payload["text"] for packet in plain.trace.packets
                                if packet.payload.get("capability") == "expert.review"
                                and packet.head.source == "RetrievalAgent")
            with session(root / "b", shared_medicines=True) as shared:
                observed = []
                deliver = shared._deliver

                def capture(kind, source, capability, text, correlation_id=None,
                            binary=None, shared_fields=None):
                    if capability == "expert.review":
                        body = json.loads(text)
                        ref = body.pop("shared_medicine_ref")
                        data, _copy_ns = read_reference(ref, shared.runtime.trace_id)
                        body["medicine_introductions"] = json.loads(data)
                        observed.append((ref, json.dumps(body, ensure_ascii=False)))
                    return deliver(kind, source, capability, text,
                                   correlation_id, binary, shared_fields)

                with patch.object(shared, "_deliver", side_effect=capture):
                    self.assertEqual(shared.run_case(case), expected)
                self.assertEqual(observed[0][1], original)
                self.assertEqual(shared.runtime.shared_memory_active_bytes, 0)
                stats = shared.runtime.shared_memory_case_stats[-1]
                self.assertEqual(stats["forwarded_write_bytes"],
                                 stats["forwarded_read_bytes"])
                self.assertGreater(stats["forwarded_write_bytes"], 0)
                with self.assertRaises(FileNotFoundError):
                    attach_shared_memory(observed[0][0]["name"])

    def test_bad_reference_aborts_case_and_next_case_runs(self):
        cases = load_dialmed_cases(DIALMED_TRAIN_PATH, 2)
        with tempfile.TemporaryDirectory() as temporary:
            with session(Path(temporary), shared_medicines=True) as shared:
                captured = []
                deliver = shared._deliver

                def corrupt(kind, source, capability, text, correlation_id=None,
                            binary=None, shared_fields=None):
                    if capability == "expert.review" and not captured:
                        body = json.loads(text)
                        captured.append(body["shared_medicine_ref"]["name"])
                        body["shared_medicine_ref"]["sha256"] = "0" * 64
                        text = json.dumps(body, ensure_ascii=False)
                    return deliver(kind, source, capability, text,
                                   correlation_id, binary, shared_fields)

                with patch.object(shared, "_deliver", side_effect=corrupt):
                    with self.assertRaisesRegex(RuntimeError, "共享药物说明引用"):
                        shared.run_case(cases[0])
                self.assertEqual(shared.runtime.shared_memory_active_bytes, 0)
                self.assertEqual(shared.runtime.state, IpcRuntimeState.READY)
                with self.assertRaises(FileNotFoundError):
                    attach_shared_memory(captured[0])
                self.assertEqual(shared.run_case(cases[1])["case_id"], cases[1].case_id)

    def test_expert_crash_reclaims_segment_and_rebuilt_session_runs(self):
        cases = load_dialmed_cases(DIALMED_TRAIN_PATH, 2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared = session(root, shared_medicines=True)
            captured = []
            deliver = shared._deliver

            def crash(kind, source, capability, text, correlation_id=None,
                      binary=None, shared_fields=None):
                if capability == "expert.review":
                    captured.append(json.loads(text)["shared_medicine_ref"]["name"])
                    process = shared.runtime._workers["ExpertAgent"][0]
                    process.terminate()
                    process.join()
                return deliver(kind, source, capability, text,
                               correlation_id, binary, shared_fields)

            try:
                with patch.object(shared, "_deliver", side_effect=crash):
                    with self.assertRaisesRegex(RuntimeError, "已退出"):
                        shared.run_case(cases[0])
                self.assertEqual(shared.runtime.state, IpcRuntimeState.FAILED)
                self.assertEqual(shared.runtime.shared_memory_active_bytes, 0)
                with self.assertRaises(FileNotFoundError):
                    attach_shared_memory(captured[0])
            finally:
                shared.close()
            with session(root, shared_medicines=True) as rebuilt:
                self.assertEqual(rebuilt.run_case(cases[1])["case_id"], cases[1].case_id)


if __name__ == "__main__":
    unittest.main()
