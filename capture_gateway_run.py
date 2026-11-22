"""Run one real, read-only hospital case and save its complete interaction trace.

This is an opt-in diagnostic command.  It records model prompts and replies,
so its output should be treated as a local case record.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from time import time_ns

from cases import DIALMED_TEST_PATH, load_dialmed_cases
from decision_vector import DecisionVectorCodec
from hospital_process import ProcessHospitalSession
from ipc_runtime import BinaryReply
from tool.llm_client import LLMClient


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
DATASET = DIALMED_TEST_PATH
REPORT = RUNS / "real_gateway_trace.json"
MARKDOWN = RUNS / "real_gateway_trace.md"
CALL_PREFIX = "real_gateway_call_"


class RecordingLLMClient(LLMClient):
    """Capture gateway payloads without recording the authentication header."""

    def chat(self, messages: list[dict[str, str]]) -> str:
        record = {"time_ns": time_ns(), "pid": os.getpid(), "model": self.model,
                  "messages": messages}
        try:
            reply = super().chat(messages)
            record["reply"] = reply
            record["usage"] = vars(self.last_usage)
            return reply
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record["finished_ns"] = time_ns()
            path = RUNS / f"{CALL_PREFIX}{os.getpid()}.jsonl"
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_markdown(report: dict) -> None:
    """Render every recorded frame and gateway payload without abbreviating it."""
    def block(value) -> str:
        return "```json\n" + json.dumps(value, ensure_ascii=False, indent=2) + "\n```\n"

    lines = ["# Tiny Hospital 单例真实网关交互记录\n",
             f"数据集：`{report['dataset']}`；病例：{report['case_id']}；"
             f"模式：四进程 + 共享对话；训练写入：{report['training']}。\n",
             "时间戳为 Unix 纳秒。`case.complete` 由本地主进程 End capability 消费，"
             "因此没有 Agent 子进程回复。网关认证头未记录。\n",
             "## 病例原始输入\n", block(report["case_input"])]
    if report.get("startup_packets"):
        lines.append("## Agent 注册握手（TH-ACP）\n")
        if report.get("startup_packets_provenance"):
            lines.append(report["startup_packets_provenance"] + "\n")
        for packet in report["startup_packets"]:
            lines.extend([f"### message_id={packet['head']['message_id']} "
                          f"{packet['head']['message_type']}\n", block(packet)])
    lines.append("## 按实际顺序记录的病例交互\n")
    for index, event in enumerate(report["events"], 1):
        if event["type"] == "control":
            lines.append(f"### {index}. 控制帧：{event['agent']} / {event['operation']}\n")
            lines.append(block({key: value for key, value in event.items()
                                if key not in ("type", "agent", "operation")}))
            continue
        lines.append(f"### {index}. 业务投递：{event['capability']}\n")
        for packet in event["packets"]:
            head = packet["head"]
            lines.append(f"#### {head['message_type']} message_id={head['message_id']} "
                         f"correlation_id={head['correlation_id']}\n")
            lines.append(block(packet))
            text = packet["payload"].get("text")
            if isinstance(text, str):
                try:
                    decoded = json.loads(text)
                except ValueError:
                    pass
                else:
                    lines.extend(["业务正文解析结果（便于阅读）：\n", block(decoded)])
        for key in ("inbound_binary", "outbound_binary"):
            if key in event:
                lines.extend([f"#### {key}\n", block(event[key])])
        lines.extend(["传输帧与共享引用：\n", block({
            "transport": event["transport"],
            "shared_references": event["shared_references"],
            "error": event.get("error"),
        })])
    lines.append("## 真实模型网关请求与回复（未省略提示词）\n")
    for index, call in enumerate(report["gateway_calls"], 1):
        lines.extend([f"### 第 {index} 次调用\n", block(call)])
    lines.extend(["## 结果与资源清理\n", block({
        "outcome": report["outcome"], "error": report["error"],
        "shared_case_statistics": report["shared_case_statistics"],
        "processes_alive_after_close": report["processes_alive_after_close"],
    })])
    MARKDOWN.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    RUNS.mkdir(exist_ok=True)
    case = load_dialmed_cases(DATASET, 1)[0]
    events: list[dict] = []
    outcome = None
    failure = None
    session = None
    startup_packets = []
    try:
        session = ProcessHospitalSession(
            retrieval_mode='jaccard', verbose=False, dataset_path=DATASET, training=False,
            llm_client_factory=RecordingLLMClient, shared_dialog=True,
            timeout=180.0, startup_timeout=30.0,
        )
        runtime = session.runtime
        startup_packets = [json.loads(packet.to_json())
                           for packet in session.trace.packets]
        original_control = runtime.control_agent
        original_deliver = session._deliver
        codec = DecisionVectorCodec()

        def capture_control(name, op, payload=None):
            event = {"type": "control", "agent": name, "operation": op,
                     "sent_payload": payload, "time_ns": time_ns()}
            events.append(event)
            try:
                event["result"] = original_control(name, op, payload)
                return event["result"]
            except Exception as exc:
                event["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                event["finished_ns"] = time_ns()

        def capture_deliver(kind, source, capability, text, correlation_id=None,
                            binary=None, shared_fields=None):
            event = {"type": "delivery", "capability": capability,
                     "time_ns": time_ns()}
            events.append(event)
            packet_start = len(session.trace.packets)
            frame_start = len(runtime.transport_frame_stats)
            if binary is not None:
                event["outbound_binary"] = {
                    "hex": binary.hex(), "length": len(binary),
                    "decoded_recommended_drugs": codec.decode(binary),
                }
            try:
                packet, result = original_deliver(
                    kind, source, capability, text, correlation_id,
                    binary, shared_fields)
                if isinstance(result, BinaryReply):
                    event["inbound_binary"] = {
                        "hex": result.binary.hex(), "length": len(result.binary),
                        "decoded_recommended_drugs": codec.decode(result.binary),
                    }
                return packet, result
            except Exception as exc:
                event["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                event["packets"] = [json.loads(packet.to_json())
                                    for packet in session.trace.packets[packet_start:]]
                event["transport"] = runtime.transport_frame_stats[frame_start:]
                event["shared_references"] = [dict(ref)
                    for _segment, ref in runtime._shared_arena._segments.values()]
                event["finished_ns"] = time_ns()

        runtime.control_agent = capture_control
        session._deliver = capture_deliver
        outcome = session.run_case(case)
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        if session is not None:
            session.close()

    gateway_calls = []
    for path in RUNS.glob(f"{CALL_PREFIX}*.jsonl"):
        with path.open(encoding="utf-8") as stream:
            gateway_calls.extend(json.loads(line) for line in stream)
        path.unlink()
    gateway_calls.sort(key=lambda row: row["time_ns"])
    report = {
        "dataset": str(DATASET), "case_id": case.case_id,
        "runtime_mode": "process", "shared_dialog": True,
        "training": False, "model": LLMClient().model,
        "case_input": {"patient_dialog": case.patient_input,
                       "actual_drugs": list(case.drug_labels)},
        "events": events, "gateway_calls": gateway_calls,
        "startup_packets": startup_packets,
        "outcome": outcome, "error": failure,
        "shared_case_statistics": (session.runtime.shared_memory_case_stats
                                   if session is not None else []),
        "processes_alive_after_close": (
            {name: process.is_alive() for name, (process, *_rest)
             in session.runtime._workers.items()} if session is not None else {}),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    write_markdown(report)
    print(json.dumps({"report": str(REPORT), "case_id": case.case_id,
                      "business_deliveries": sum(event["type"] == "delivery"
                                                 for event in events),
                      "gateway_calls": len(gateway_calls),
                      "outcome": outcome, "error": failure}, ensure_ascii=False))
    if failure:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
