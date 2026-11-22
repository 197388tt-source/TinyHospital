"""Communication trace and per-case packet usage statistics."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter_ns

from protocol import AgentPacket, MessageType


@dataclass(frozen=True)
class MessageCost:
    step: int
    message_id: int
    capability: str
    sender: str
    receiver: str
    characters: int
    utf8_bytes: int
    packet_utf8_bytes: int
    message_tokens_estimate: int
    packet_tokens_estimate: int
    execution_ns: int = 0
    serialization_ns: int = 0


@dataclass(frozen=True)
class PacketCost:
    trace_id: int
    message_id: int
    step: int | None
    message_type: str
    source: str
    target: str | None
    capability: str | None
    message_bytes: int
    packet_bytes: int
    message_tokens_estimate: int
    packet_tokens_estimate: int
    execution_ns: int = 0
    serialization_ns: int = 0


@dataclass(frozen=True)
class CaseCost:
    trace_id: int
    message_count: int
    message_bytes: int
    packet_bytes: int
    message_tokens_estimate: int
    packet_tokens_estimate: int
    execution_ns: int = 0
    serialization_ns: int = 0


def estimate_tokens(value: str) -> int:
    """A stable communication estimate, not a model tokenizer count."""
    return len(re.findall(r"[\u3400-\u9fff]|[A-Za-z]+|\d+|[^\s]", value))


def append_communication_records(path: Path, costs: list[PacketCost], case_id: int | None = None) -> None:
    if not costs:
        return
    scope = "control" if costs[0].trace_id == 0 else "case"
    trace_id = costs[0].trace_id
    if any(cost.trace_id != trace_id or cost.target is None for cost in costs):
        raise RuntimeError("通信报文缺少统一 trace_id 或目标 Agent")
    rows = [
        {
            "record_type": "packet",
            "scope": scope,
            "case_id": case_id,
            "trace_id": cost.trace_id,
            "message_id": cost.message_id,
            "step": cost.step,
            "message_type": cost.message_type,
            "source": cost.source,
            "target": cost.target,
            "capability": cost.capability,
            "message_bytes": cost.message_bytes,
            "packet_bytes": cost.packet_bytes,
            "message_tokens_estimate": cost.message_tokens_estimate,
            "packet_tokens_estimate": cost.packet_tokens_estimate,
            "execution_ns": cost.execution_ns,
            "serialization_ns": cost.serialization_ns,
        }
        for cost in costs
    ]
    rows.append({
        "record_type": f"{scope}_summary",
        "scope": scope,
        "case_id": case_id,
        "trace_id": trace_id,
        "message_count": len(costs),
        "total_message_bytes": sum(cost.message_bytes for cost in costs),
        "total_packet_bytes": sum(cost.packet_bytes for cost in costs),
        "total_message_tokens_estimate": sum(cost.message_tokens_estimate for cost in costs),
        "total_packet_tokens_estimate": sum(cost.packet_tokens_estimate for cost in costs),
        "total_execution_ns": sum(cost.execution_ns for cost in costs),
        "total_serialization_ns": sum(cost.serialization_ns for cost in costs),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.writelines(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        )


@dataclass
class Trace:
    verbose: bool = True
    messages: list[tuple[str, str, str]] = field(default_factory=list)
    message_costs: list[MessageCost] = field(default_factory=list)
    packets: list[AgentPacket] = field(default_factory=list)
    packet_costs: list[PacketCost] = field(default_factory=list)
    states: list[tuple[str, str, str]] = field(default_factory=list)
    case_costs: list[CaseCost] = field(default_factory=list)
    _case_message_start: int | None = None
    _case_packet_start: int | None = None

    def begin_case(self, trace_id: int) -> None:
        if self._case_message_start is not None:
            raise RuntimeError("上一病例的统计尚未结束")
        self._case_message_start = len(self.messages)
        self._case_packet_start = len(self.packets)
        if self.verbose:
            print(f"\n=== 开始病例 trace_id={trace_id} ===")

    def packet(self, packet: AgentPacket) -> int:
        is_flow = packet.head.message_type in (MessageType.REQUEST.value, MessageType.RESPONSE.value)
        step = len(self.messages) - self._case_message_start if is_flow and self._case_message_start is not None else None
        serialization_started_ns = perf_counter_ns()
        wire = packet.to_json()
        trace_serialization_ns = perf_counter_ns() - serialization_started_ns
        body = packet.payload.get("text")
        if body is None:
            body = json.dumps(
                packet.payload.get("data", packet.payload.get("parameters", packet.payload)),
                ensure_ascii=False, separators=(",", ":"), sort_keys=True,
            )
        kind = packet.head.message_type
        target = "Runtime" if kind == MessageType.HELLO.value else None
        if kind in (MessageType.HELLO_ACK.value, MessageType.ERROR.value):
            parent = next((
                previous for previous in reversed(self.packets)
                if previous.head.trace_id == packet.head.trace_id
                and previous.head.message_id == packet.head.correlation_id
            ), None)
            target = parent.head.source if parent is not None else None
        self.packets.append(packet)
        capability = packet.payload.get("capability")
        cost = PacketCost(
            packet.head.trace_id, packet.head.message_id, step, kind, packet.head.source,
            target, capability, len(body.encode("utf-8")), len(wire.encode("utf-8")),
            estimate_tokens(body), estimate_tokens(wire),
            serialization_ns=trace_serialization_ns,
        )
        self.packet_costs.append(cost)
        if self.verbose:
            label = f"第{step}步" if step is not None else "控制"
            destination = f"capability={capability}" if capability else "Runtime/控制"
            print(
                f"[报文开销] {label} {kind} "
                f"{packet.head.source} -> {destination}: {cost.packet_bytes} UTF-8 字节"
            )
        return len(self.packet_costs) - 1

    def finish_delivery(self, packet_index: int, execution_ns: int,
                        serialization_ns: int) -> None:
        """Record the whole synchronous delivery and its JSON boundary work."""
        cost = self.packet_costs[packet_index]
        cost = replace(cost, execution_ns=execution_ns,
                       serialization_ns=cost.serialization_ns + serialization_ns)
        self.packet_costs[packet_index] = cost
        if cost.step is not None:
            message_index = next((index for index in range(len(self.message_costs) - 1, -1, -1)
                                  if self.message_costs[index].message_id == cost.message_id
                                  and self.message_costs[index].step == cost.step), None)
            if message_index is not None:
                self.message_costs[message_index] = replace(
                    self.message_costs[message_index], execution_ns=cost.execution_ns,
                    serialization_ns=cost.serialization_ns,
                )

    def state(self, agent: str, before: str, after: str) -> None:
        self.states.append((agent, before, after))
        if self.verbose:
            print(f"[状态] {agent}: {before} -> {after}")

    def message(self, sender: str, receiver: str, body: str) -> None:
        # Step 0 is the System kickoff; subsequent steps follow the active route.
        if self._case_message_start is None:
            raise RuntimeError("尚未开始病例统计")
        step = len(self.messages) - self._case_message_start
        self.messages.append((sender, receiver, body))
        packet = self.packets[-1]
        self.packet_costs[-1] = replace(self.packet_costs[-1], target=receiver)
        packet_cost = self.packet_costs[-1]
        cost = MessageCost(
            step, packet.head.message_id, packet.payload["capability"], sender, receiver,
            len(body), packet_cost.message_bytes, packet_cost.packet_bytes,
            packet_cost.message_tokens_estimate, packet_cost.packet_tokens_estimate,
        )
        self.message_costs.append(cost)
        if self.verbose:
            print(f"[通信] {sender} -> {receiver}: {body}")
            label = "启动" if step == 0 else f"第{step}步"
            print(
                f"[通信开销] {label}: {cost.characters} 字符，{cost.utf8_bytes} UTF-8 字节，"
                f"正文约 {cost.message_tokens_estimate} token，报文约 {cost.packet_tokens_estimate} token"
            )

    def finish_case(self, trace_id: int) -> CaseCost:
        if self._case_message_start is None or self._case_packet_start is None:
            raise RuntimeError("尚未开始病例统计")
        case_packets = self.packet_costs[self._case_packet_start:]
        summary = CaseCost(
            trace_id, len(case_packets),
            sum(cost.message_bytes for cost in case_packets),
            sum(cost.packet_bytes for cost in case_packets),
            sum(cost.message_tokens_estimate for cost in case_packets),
            sum(cost.packet_tokens_estimate for cost in case_packets),
            sum(cost.execution_ns for cost in case_packets),
            sum(cost.serialization_ns for cost in case_packets),
        )
        self.case_costs.append(summary)
        self._case_message_start = None
        self._case_packet_start = None
        if not self.verbose:
            return summary
        print(f"\n=== 病例 {trace_id} 通信开销（含 System 启动和 End 通知） ===")
        for cost in case_packets:
            print(
                f"message_id={cost.message_id} {cost.source} -> {cost.target}: "
                f"正文 {cost.message_bytes} 字节 / 约 {cost.message_tokens_estimate} token，"
                f"报文 {cost.packet_bytes} 字节 / 约 {cost.packet_tokens_estimate} token，"
                f"执行 {cost.execution_ns / 1_000_000:.3f} ms，"
                f"序列化 {cost.serialization_ns / 1_000_000:.3f} ms"
            )
        print(
            f"病例合计：{summary.message_count} 条，"
            f"正文 {summary.message_bytes} 字节 / 约 {summary.message_tokens_estimate} token，"
            f"报文 {summary.packet_bytes} 字节 / 约 {summary.packet_tokens_estimate} token，"
            f"执行 {summary.execution_ns / 1_000_000:.3f} ms，"
            f"序列化 {summary.serialization_ns / 1_000_000:.3f} ms"
        )
        return summary

    def print_overall_summary(self) -> None:
        control = [cost for cost in self.packet_costs if cost.trace_id == 0]
        case_count = len(self.case_costs)
        if case_count == 0:
            return
        case_bytes = sum(cost.packet_bytes for cost in self.case_costs)
        case_tokens = sum(cost.packet_tokens_estimate for cost in self.case_costs)
        print("\n=== 多病例通信开销汇总 ===")
        print(
            f"控制报文：{len(control)} 条，"
            f"{sum(cost.packet_bytes for cost in control)} 字节，"
            f"约 {sum(cost.packet_tokens_estimate for cost in control)} token"
        )
        print(
            f"病例业务报文：{sum(cost.message_count for cost in self.case_costs)} 条，"
            f"{case_bytes} 字节，约 {case_tokens} token"
        )
        print(
            f"平均每病例：{case_bytes / case_count:.1f} 字节，"
            f"约 {case_tokens / case_count:.1f} 通信估算 token，"
            f"执行 {sum(cost.execution_ns for cost in self.case_costs) / case_count / 1_000_000:.3f} ms，"
            f"序列化 {sum(cost.serialization_ns for cost in self.case_costs) / case_count / 1_000_000:.3f} ms"
        )
        print("执行耗时包含 Handler 与模型 API 等待；通信 token 为字符规则估算，模型 API 实际 token 单独记录。")
