"""Measure the Doctor→Expert handoff for ten deterministic cases.

Counts the complete Pipe frames for the Doctor reply and Expert request only.
The JSON comparison rebuilds those two frames with the same TH-ACP heads.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from statistics import mean
from unittest.mock import patch

from cases import DIALMED_TRAIN_PATH, load_dialmed_cases
from hospital_process import ProcessHospitalSession
from main import HospitalSession
from performance_baseline import FixedModel
from protocol import AgentPacket


def _paths(root: Path) -> dict:
    return {
        "golden_path": root / "golden.json",
        "reflection_path": root / "reflections.json",
        "communication_path": root / "communications.jsonl",
        "control_path": root / "control.jsonl",
        "llm_path": root / "llm.jsonl",
        "dataset_path": DIALMED_TRAIN_PATH,
    }


def measure() -> dict:
    cases = load_dialmed_cases(DIALMED_TRAIN_PATH, 10)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        with patch("main.LLMClient", side_effect=FixedModel):
            sync = HospitalSession(retrieval_mode='jaccard', verbose=False, training=False, **_paths(root / "sync"))
            try:
                expected = [sync.run_case(case) for case in cases]
            finally:
                sync.close()
        original = {
            json.loads(packet.payload["text"])["case_id"]: packet.payload["text"]
            for packet in sync.trace.packets
            if packet.payload.get("capability") == "expert.evaluate"
            and packet.head.source == "DoctorAgent"
        }
        with ProcessHospitalSession(
            retrieval_mode='jaccard', verbose=False, training=False, llm_client_factory=FixedModel,
            **_paths(root / "process"),
        ) as process:
            actual = [process.run_case(case) for case in cases]
            assert actual == expected
            rows = list(process.runtime.binary_frame_stats)
            codec = list(process.decision_codec_stats)
            relevant = [packet for packet in process.trace.packets
                        if ((packet.payload.get("capability") == "doctor.recommend"
                             and packet.head.source == "DoctorAgent")
                            or (packet.payload.get("capability") == "expert.evaluate"
                                and packet.head.source == "DoctorAgent"))]
        assert len(rows) == len(relevant) == len(codec) == 20
        json_frame_bytes = 0
        for packet in relevant:
            case_id = json.loads(packet.payload["text"])["case_id"]
            full = AgentPacket(packet.head, {
                "capability": packet.payload["capability"], "text": original[case_id],
            })
            # Each frame has a 4-byte length; requests also reserve an 8-byte reply ID.
            json_frame_bytes += 4 + len(full.to_json().encode("utf-8"))
            if packet.payload["capability"] == "expert.evaluate":
                json_frame_bytes += 8
        actual_frame_bytes = sum(row["frame_bytes"] for row in rows)
        return {
            "cases": len(cases),
            "handoff_frames": len(rows),
            "outputs_identical": actual == expected,
            "json_frames_bytes_reconstructed": json_frame_bytes,
            "binary_frames_bytes_measured": actual_frame_bytes,
            "saved_bytes": json_frame_bytes - actual_frame_bytes,
            "saved_percent": round(100 * (json_frame_bytes - actual_frame_bytes) / json_frame_bytes, 2),
            "raw_binary_payload_bytes": sum(row["binary_bytes"] for row in rows),
            "encode_mean_us": round(mean(row["codec_ns"] for row in codec
                                         if row["operation"] == "encode") / 1000, 2),
            "decode_mean_us": round(mean(row["codec_ns"] for row in codec
                                         if row["operation"] == "decode") / 1000, 2),
            "scope": "Doctor response + Expert request; Pipe frame headers included; no LLM timing",
        }


if __name__ == "__main__":
    result = measure()
    output = Path(__file__).resolve().parent / "runs/stage6_decision_transfer.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
