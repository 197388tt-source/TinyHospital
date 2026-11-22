"""TH-ACP packets and the local, synchronous Agent runtime."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from threading import Lock
from time import perf_counter_ns
from typing import Any, Callable

VERSION = "TH-ACP/1.0"
RETRIEVAL_CAPABILITY = "retrieval.retrieve"


class LocalIds:
    """Process-local case IDs and per-trace packet sequence numbers."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._next_trace = 1
        self._message_numbers: dict[int, int] = {}

    def new_trace(self) -> int:
        with self._lock:
            trace_id = self._next_trace
            self._next_trace += 1
            return trace_id

    def next_message(self, trace_id: int) -> int:
        if type(trace_id) is not int or trace_id < 0:
            raise ValueError("trace_id 必须是非负整数")
        with self._lock:
            message_id = self._message_numbers.get(trace_id, 0) + 1
            self._message_numbers[trace_id] = message_id
            return message_id


LOCAL_IDS = LocalIds()


class MessageType(str, Enum):
    HELLO = "HELLO"
    HELLO_ACK = "HELLO_ACK"
    REQUEST = "REQUEST"
    RESPONSE = "RESPONSE"
    ERROR = "ERROR"


@dataclass(frozen=True)
class PacketHead:
    version: str
    message_id: int
    trace_id: int
    message_type: str
    source: str
    correlation_id: int | None


@dataclass(frozen=True)
class AgentPacket:
    head: PacketHead
    payload: dict[str, Any]

    @classmethod
    def create(
        cls, *, trace_id: int, message_type: MessageType, source: str,
        payload: dict[str, Any] | None = None, correlation_id: int | None = None,
    ) -> AgentPacket:
        head = PacketHead(VERSION, LOCAL_IDS.next_message(trace_id), trace_id, message_type.value, source, correlation_id)
        return cls(head, payload if payload is not None else {})

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":"), sort_keys=True)


class ProtocolParser:
    """Parse a complete packet; malformed input becomes an ERROR packet."""

    HEAD_FIELDS = set(PacketHead.__dataclass_fields__)
    PAYLOAD_FIELDS = {
        MessageType.HELLO.value: {"capabilities"},
        MessageType.HELLO_ACK.value: {"status", "capabilities"},
        MessageType.REQUEST.value: {"capability", "text"},
        MessageType.RESPONSE.value: {"capability", "text"},
        MessageType.ERROR.value: {"error"},
    }

    @staticmethod
    def _json_value(value: Any) -> bool:
        if value is None or type(value) in (str, bool, int):
            return True
        if type(value) is float:
            return math.isfinite(value)
        if isinstance(value, list):
            return all(ProtocolParser._json_value(item) for item in value)
        if isinstance(value, dict):
            return all(isinstance(key, str) and ProtocolParser._json_value(item)
                       for key, item in value.items())
        return False

    def parse(self, raw: str | bytes | dict[str, Any] | AgentPacket) -> AgentPacket:
        data: Any = None
        try:
            data = json.loads(raw) if isinstance(raw, (str, bytes)) else asdict(raw) if isinstance(raw, AgentPacket) else raw
            self._validate(data)
            return AgentPacket(PacketHead(**data["head"]), data["payload"])
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            head = data.get("head") if isinstance(data, dict) else None
            head = head if isinstance(head, dict) else {}
            valid_trace = self._integer(head.get("trace_id"), minimum=0)
            return AgentPacket.create(
                trace_id=head["trace_id"] if valid_trace else 0,
                message_type=MessageType.ERROR, source="Runtime", payload={"error": str(exc)},
                correlation_id=head["message_id"] if valid_trace and self._integer(head.get("message_id"), minimum=1) else None,
            )

    @staticmethod
    def _integer(value: Any, *, minimum: int) -> bool:
        return type(value) is int and value >= minimum

    def _validate(self, data: Any) -> None:
        if not isinstance(data, dict):
            raise ValueError("报文必须是 JSON 对象")
        if set(data) != {"head", "payload"}:
            raise ValueError("报文只允许 head 和 payload")
        head, payload = data["head"], data["payload"]
        if not isinstance(head, dict) or not isinstance(payload, dict):
            raise ValueError("head 和 payload 必须是对象")
        missing = self.HEAD_FIELDS - head.keys()
        if missing:
            raise ValueError(f"head 缺少字段：{', '.join(sorted(missing))}")
        if set(head) != self.HEAD_FIELDS:
            raise ValueError("head 含不支持的字段")
        if head["version"] != VERSION:
            raise ValueError("不支持的协议版本")
        if not self._integer(head["message_id"], minimum=1):
            raise ValueError("message_id 必须是正整数")
        if not self._integer(head["trace_id"], minimum=0):
            raise ValueError("trace_id 必须是非负整数")
        if head["correlation_id"] is not None and not self._integer(head["correlation_id"], minimum=1):
            raise ValueError("correlation_id 必须是正整数或 null")
        if head["message_type"] not in self.PAYLOAD_FIELDS:
            raise ValueError("未知 message_type")
        if not isinstance(head["source"], str) or not head["source"].strip():
            raise ValueError("source 必须是非空字符串")
        kind = head["message_type"]
        if kind in (MessageType.HELLO.value, MessageType.HELLO_ACK.value) and head["trace_id"] != 0:
            raise ValueError(f"{kind} 必须使用 trace_id=0")
        if kind in (MessageType.REQUEST.value, MessageType.RESPONSE.value) and head["trace_id"] == 0:
            raise ValueError(f"{kind} 必须使用病例 trace_id")
        expected = (
            {"capability", "parameters"}
            if kind == MessageType.REQUEST.value and payload.get("capability") == RETRIEVAL_CAPABILITY
            else ({"capability", "data"} if kind in (MessageType.REQUEST.value, MessageType.RESPONSE.value)
                  and "data" in payload else self.PAYLOAD_FIELDS[kind])
        )
        if set(payload) != expected:
            raise ValueError(f"{kind} 的 payload 必须且只能包含 {', '.join(sorted(expected)) or '空对象'}")
        if kind == MessageType.HELLO_ACK.value:
            if payload["status"] != "ok":
                raise ValueError(f"{kind} 的 status 必须为 ok")
        if kind in (MessageType.HELLO.value, MessageType.HELLO_ACK.value):
            capabilities = payload["capabilities"]
            if not isinstance(capabilities, list) or not capabilities or any(
                not isinstance(item, str) or not item.strip() for item in capabilities
            ) or len(capabilities) != len(set(capabilities)):
                raise ValueError(f"{kind} 需要无重复的 capabilities 列表")
        if kind in (MessageType.REQUEST.value, MessageType.RESPONSE.value):
            if not isinstance(payload["capability"], str) or not payload["capability"].strip():
                raise ValueError(f"{kind} 需要 capability")
            if kind == MessageType.REQUEST.value and payload["capability"] == RETRIEVAL_CAPABILITY:
                parameters = payload["parameters"]
                required = {"candidate_diseases", "knowledge_type", "include_history"}
                if not isinstance(parameters, dict) or not required <= set(parameters) or set(parameters) - required - {"memory_types"}:
                    raise ValueError("检索 REQUEST 的 parameters 字段不完整或包含多余字段")
                candidates = parameters["candidate_diseases"]
                if not isinstance(candidates, list) or not 1 <= len(candidates) <= 3 or any(
                    not isinstance(name, str) or not name.strip() for name in candidates
                ) or len(candidates) != len(set(candidates)):
                    raise ValueError("candidate_diseases 必须是 1～3 个无重复疾病名称")
                if parameters["knowledge_type"] != "symptoms":
                    raise ValueError("knowledge_type 当前只支持 symptoms")
                if type(parameters["include_history"]) is not bool:
                    raise ValueError("include_history 必须是布尔值")
                memory_types = parameters.get("memory_types", ["case", "experience"])
                if not isinstance(memory_types, list) or not memory_types or any(
                    item not in ("case", "experience") for item in memory_types
                ) or len(memory_types) != len(set(memory_types)):
                    raise ValueError("memory_types 必须选择不重复的 case 和/或 experience")
            elif "data" in payload:
                if not isinstance(payload["data"], dict) or not self._json_value(payload["data"]):
                    raise ValueError(f"{kind} 需要 JSON 对象 data")
            elif not isinstance(payload["text"], str):
                raise ValueError(f"{kind} 需要 text 字符串")
            if payload['capability'] in ('patient.request_history', 'retrieval.history'):
                body = payload.get('data')
                if body is None and isinstance(payload.get('text'), str):
                    try:
                        body = json.loads(payload['text'])
                    except ValueError:
                        body = None  # System's natural-language Patient request.
                if isinstance(body, dict) and 'query_embedding' in body:
                    validate_embedding_descriptor(body['query_embedding'], head['trace_id'])
        if kind == MessageType.ERROR.value and (
            not isinstance(payload["error"], str) or not payload["error"].strip()
        ):
            raise ValueError("ERROR 需要 error 字符串")
        if kind in (MessageType.HELLO_ACK.value, MessageType.RESPONSE.value) and not head["correlation_id"]:
            raise ValueError(f"{kind} 需要 correlation_id")

    @staticmethod
    def extract_capability(packet: AgentPacket) -> str:
        return packet.payload["capability"]


class CapabilityRegistry:
    def __init__(self) -> None:
        self._owners: dict[str, str] = {}

    def register(self, agent_name: str, capability: str) -> None:
        owner = self._owners.get(capability)
        if owner is not None and owner != agent_name:
            raise ValueError(f"capability {capability} 已由 {owner} 注册")
        self._owners[capability] = agent_name

    def resolve(self, capability: str) -> str | None:
        return self._owners.get(capability)


class ActionRegistry:
    """Map a registered capability directly to its handler."""

    def __init__(self) -> None:
        self._handlers: dict[str, Callable[[Any], Any]] = {}

    def register(self, capability: str, handler: Callable[[Any], Any]) -> None:
        if capability in self._handlers:
            raise ValueError(f"capability {capability} 的 Handler 重复注册")
        self._handlers[capability] = handler

    def resolve(self, capability: str) -> Callable[[Any], Any] | None:
        return self._handlers.get(capability)


@dataclass(frozen=True)
class Delivery:
    packet: AgentPacket
    output: Any = None


class Runtime:
    def __init__(self, trace: Any) -> None:
        self.trace = trace
        self.trace_id: int | None = None
        self.parser = ProtocolParser()
        self.capabilities = CapabilityRegistry()
        self.actions = ActionRegistry()
        self.agents: dict[str, Any] = {}
        self._pending: dict[str, tuple[Any, dict[str, Callable[[Any], Any]]]] = {}
        self.capabilities.register("End", "case.complete")
        self.actions.register("case.complete", lambda _text: None)

    def start_case(self) -> int:
        if self.trace_id is not None:
            raise RuntimeError("当前病例尚未结束")
        self.trace_id = LOCAL_IDS.new_trace()
        return self.trace_id

    def finish_case(self) -> None:
        if self.trace_id is None:
            raise RuntimeError("当前没有运行中的病例")
        self.trace_id = None

    def register_agent(self, agent: Any, handlers: dict[str, Callable[[Any], Any]]) -> None:
        if agent.name in self.agents or agent.name in self._pending:
            raise ValueError(f"Agent {agent.name} 重复注册")
        if not handlers:
            raise ValueError("Agent 至少要注册一种 capability")
        agent.runtime = self
        self._pending[agent.name] = (agent, handlers)
        try:
            hello = agent.make_packet(MessageType.HELLO, payload={"capabilities": list(handlers)})
            delivery = self.deliver(hello)
            if delivery.packet.head.message_type == MessageType.ERROR.value:
                raise RuntimeError(delivery.packet.payload["error"])
            ack = delivery.output
            if (
                not isinstance(ack, AgentPacket)
                or ack.head.message_type != MessageType.HELLO_ACK.value
                or ack.head.correlation_id != hello.head.message_id
                or ack.payload["capabilities"] != list(handlers)
            ):
                raise RuntimeError("HELLO 握手或 capability 注册失败")
        except Exception:
            agent.runtime = None
            raise
        finally:
            self._pending.pop(agent.name, None)

    def deliver(self, raw: str | bytes | dict[str, Any] | AgentPacket) -> Delivery:
        # Always cross the serialized protocol boundary before dispatch.
        started_ns = perf_counter_ns()
        serialization_started_ns = perf_counter_ns()
        if isinstance(raw, AgentPacket):
            try:
                wire = raw.to_json()
            except (TypeError, ValueError):
                # Let the parser return its correlated ERROR for invalid data
                # whose keys cannot be sorted or whose values cannot be encoded.
                wire = raw
        else:
            wire = raw
        serialization_ns = perf_counter_ns() - serialization_started_ns
        serialization_started_ns = perf_counter_ns()
        packet = self.parser.parse(wire)
        serialization_ns += perf_counter_ns() - serialization_started_ns
        packet_index = self.trace.packet(packet)
        try:
            return self._dispatch(packet)
        finally:
            self.trace.finish_delivery(packet_index, perf_counter_ns() - started_ns,
                                       serialization_ns)

    def _dispatch(self, packet: AgentPacket) -> Delivery:
        if packet.head.message_type == MessageType.ERROR.value:
            return Delivery(packet)
        kind = packet.head.message_type
        expected_trace = 0 if kind == MessageType.HELLO.value else self.trace_id
        if packet.head.trace_id != expected_trace:
            return self._error(packet, "trace_id 与当前任务不匹配")
        if kind == MessageType.HELLO.value:
            pending = self._pending.get(packet.head.source)
            if pending is None or packet.head.source in self.agents:
                return self._error(packet, "未知 Agent 或重复 HELLO")
            agent, handlers = pending
            accepted = packet.payload["capabilities"]
            if set(accepted) != set(handlers):
                return self._error(packet, "HELLO 的 capabilities 与 Agent 声明不一致")
            for capability in accepted:
                if self.capabilities.resolve(capability) is not None or self.actions.resolve(capability) is not None:
                    return self._error(packet, f"capability {capability} 已注册")
            for capability in accepted:
                self.capabilities.register(packet.head.source, capability)
                self.actions.register(capability, handlers[capability])
            self.agents[packet.head.source] = agent
            ack = AgentPacket.create(
                trace_id=0, message_type=MessageType.HELLO_ACK,
                source="Runtime", correlation_id=packet.head.message_id,
                payload={"status": "ok", "capabilities": accepted},
            )
            ack_started_ns = perf_counter_ns()
            serialization_started_ns = perf_counter_ns()
            ack_wire = ack.to_json()
            ack = self.parser.parse(ack_wire)
            ack_serialization_ns = perf_counter_ns() - serialization_started_ns
            ack_index = self.trace.packet(ack)
            self.trace.finish_delivery(ack_index, perf_counter_ns() - ack_started_ns,
                                       ack_serialization_ns)
            return Delivery(packet, ack)
        if kind in (MessageType.REQUEST.value, MessageType.RESPONSE.value):
            if packet.head.source != "System" and packet.head.source not in self.agents:
                return self._error(packet, "未知发送方 Agent")
            capability = self.parser.extract_capability(packet)
            receiver = self.capabilities.resolve(capability)
            handler = self.actions.resolve(capability)
            if receiver is None or handler is None:
                return self._error(packet, "capability 未注册或没有 Handler")
            if receiver == "End" and kind != MessageType.RESPONSE.value:
                return self._error(packet, "End 只接受 RESPONSE")
            content = packet.payload.get("parameters", packet.payload.get("data", packet.payload.get("text")))
            logged_content = (
                json.dumps(content, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
                if isinstance(content, dict) else content
            )
            self.trace.message(packet.head.source, receiver, logged_content)
            return Delivery(packet, handler(content))
        return self._error(packet, "不支持的报文类型")

    def _error(self, packet: AgentPacket, reason: str) -> Delivery:
        started_ns = perf_counter_ns()
        error = AgentPacket.create(
            trace_id=packet.head.trace_id, message_type=MessageType.ERROR,
            source="Runtime", correlation_id=packet.head.message_id,
            payload={"error": reason},
        )
        error_index = self.trace.packet(error)
        self.trace.finish_delivery(error_index, perf_counter_ns() - started_ns, 0)
        return Delivery(error)


def validate_embedding_descriptor(value: Any, trace_id: int) -> None:
    """TH-ACP semantic sidecar v1. Payload bytes are validated by the receiver."""
    import re
    required = {'version', 'kind', 'storage', 'model_fingerprint', 'dimension',
                'dtype', 'normalized', 'size', 'sha256', 'text_hash', 'trace_id'}
    if not isinstance(value, dict):
        raise ValueError('向量描述必须是对象')
    shared = value.get('storage') == 'shared_memory'
    if set(value) != required | ({'reference'} if shared else set()):
        raise ValueError('向量描述字段无效')
    if (value['version'] != 1 or type(value['version']) is not int
            or value['kind'] != 'embedding' or value['storage'] not in ('shared_memory', 'binary')
            or value['dtype'] != 'float32-le' or value['normalized'] is not True
            or type(value['dimension']) is not int or not 0 < value['dimension'] <= 65536
            or type(value['size']) is not int or value['size'] != value['dimension'] * 4
            or type(value['trace_id']) is not int or value['trace_id'] != trace_id):
        raise ValueError('向量类型、维度、长度或病例归属无效')
    for field in ('model_fingerprint', 'sha256', 'text_hash'):
        if not isinstance(value[field], str) or not re.fullmatch('[0-9a-f]{64}', value[field]):
            raise ValueError('向量指纹无效')
