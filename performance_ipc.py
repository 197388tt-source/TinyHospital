"""Deterministic, LLM-free synchronous versus Pipe roundtrip measurement."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter_ns

from ipc_runtime import MultiprocessRuntime
from protocol import AgentPacket, MessageType, Runtime
from tool.tracing import Trace

REPORT_PATH = Path(__file__).resolve().parent / "runs/performance_ipc.json"
CAPABILITY = "benchmark.echo"
TEXT = "确定性模拟 Agent：你好，Tiny Hospital。" * 4


def echo(text: str) -> str:
    return text


class BenchmarkAgent:
    name = "BenchmarkAgent"
    runtime: Runtime | None = None

    def make_packet(self, message_type: MessageType, *, payload: dict) -> AgentPacket:
        return AgentPacket.create(trace_id=0, message_type=message_type,
                                  source=self.name, payload=payload)


def _percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) * fraction)]


def _run(runtime: Runtime, kind: MessageType, *, iterations: int, warmup: int) -> dict:
    if runtime.trace_id is None:
        trace_id = runtime.start_case()
        runtime.trace.begin_case(trace_id)
    else:
        trace_id = runtime.trace_id
    timings: list[int] = []
    request_sizes: list[int] = []
    response_sizes: list[int] = []
    frame_sizes: list[int] = []
    first_roundtrip_ms = 0.0
    for index in range(warmup + iterations):
        request = AgentPacket.create(
            trace_id=trace_id, message_type=kind,
            source="System", correlation_id=1 if kind == MessageType.RESPONSE else None,
            payload={"capability": CAPABILITY, "text": TEXT},
        )
        started = perf_counter_ns()
        delivery = runtime.deliver(request)
        elapsed = perf_counter_ns() - started
        if index == 0:
            first_roundtrip_ms = round(elapsed / 1_000_000, 2)
        if delivery.output != TEXT:
            raise AssertionError("模拟 Agent 响应不一致")
        if index < warmup:
            continue
        timings.append(elapsed)
        request_sizes.append(len(request.to_json().encode("utf-8")))
        if isinstance(runtime, MultiprocessRuntime):
            response = delivery.packet
            if response.head.correlation_id != request.head.message_id:
                raise AssertionError("响应关联失败")
            response_sizes.append(len(response.to_json().encode("utf-8")))
            # 4-byte frame length + 8-byte parent-allocated response ID.
            frame_sizes.append(12 + len(request.to_json().encode("utf-8")))
    return {
        "samples": len(timings),
        "first_roundtrip_ms": first_roundtrip_ms,
        "mean_latency_us": round(mean(timings) / 1000, 2),
        "median_latency_us": round(median(timings) / 1000, 2),
        "p95_latency_us": round(_percentile(timings, 0.95) / 1000, 2),
        "mean_request_packet_bytes": round(mean(request_sizes), 2),
        "mean_response_packet_bytes": round(mean(response_sizes), 2) if response_sizes else 0,
        "mean_pipe_request_frame_bytes": round(mean(frame_sizes), 2) if frame_sizes else 0,
    }


def compare(*, iterations: int = 200, warmup: int = 20) -> dict:
    if iterations <= 0 or warmup < 0:
        raise ValueError("iterations 必须大于 0，warmup 不能为负")
    sync_trace = Trace(verbose=False)
    sync = Runtime(sync_trace)
    sync.register_agent(BenchmarkAgent(), {CAPABILITY: echo})
    sync_result = _run(sync, MessageType.REQUEST, iterations=iterations, warmup=warmup)
    sync_response_result = _run(sync, MessageType.RESPONSE, iterations=iterations, warmup=warmup)

    ipc_trace = Trace(verbose=False)
    with MultiprocessRuntime(ipc_trace, timeout=5) as ipc:
        started = perf_counter_ns()
        ipc.register_process_agent("BenchmarkAgent", CAPABILITY, echo)
        startup_ms = round((perf_counter_ns() - started) / 1_000_000, 2)
        ipc_result = _run(ipc, MessageType.REQUEST, iterations=iterations, warmup=warmup)
        ipc_response_result = _run(ipc, MessageType.RESPONSE, iterations=iterations, warmup=warmup)

    def comparison(local: dict, remote: dict) -> dict:
        return {
            "mean_extra_latency_us": round(remote["mean_latency_us"] - local["mean_latency_us"], 2),
            "mean_latency_ratio": round(remote["mean_latency_us"] / local["mean_latency_us"], 2),
            "mean_extra_th_acp_bytes": round(
                remote["mean_request_packet_bytes"] + remote["mean_response_packet_bytes"]
                - local["mean_request_packet_bytes"], 2),
            "mean_pipe_frame_overhead_bytes": round(
                remote["mean_pipe_request_frame_bytes"] - remote["mean_request_packet_bytes"], 2),
        }

    return {
        "fixture": {"iterations": iterations, "warmup": warmup,
                    "handler": "echo(text) -> text", "text_utf8_bytes": len(TEXT.encode("utf-8")),
                    "startup_excluded_from_roundtrip": True},
        "synchronous": sync_result,
        "ipc": ipc_result,
        "comparison": comparison(sync_result, ipc_result),
        "synchronous_response": sync_response_result,
        "ipc_response": ipc_response_result,
        "response_comparison": comparison(sync_response_result, ipc_response_result),
        "process_startup_and_ready_ms": startup_ms,
        "notes": [
            "同步模式解析输入报文并直接返回 Handler 输出；IPC 模式额外发送 TH-ACP RESPONSE 确认处理结果。",
            "延迟包含 Runtime 和 Trace 开销；IPC 包含 Pipe 编帧、进程调度及响应解析。",
            "启动时间包含子进程就绪确认和 HELLO 注册；均值和分位数排除了预热轮次。",
            "字节数是 UTF-8 TH-ACP 报文长度；Pipe 输入帧另有 4 字节长度和 8 字节响应 ID。",
            "模拟 Agent 不调用 LLM；时间是单机测量，不代表真实医疗流程延迟。",
        ],
    }


if __name__ == "__main__":
    report = compare()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
