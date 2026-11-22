"""Optional TH-ACP request/response transport for isolated Agent handlers.

Only explicitly registered handlers run in child processes. The normal Runtime
and the hospital's existing synchronous route remain unchanged.
"""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing import shared_memory
import json
import os
import secrets
import selectors
import struct
from dataclasses import dataclass, replace
from enum import Enum
from multiprocessing.connection import Connection
from time import monotonic, perf_counter_ns
from typing import Any, Callable

from protocol import (AgentPacket, Delivery, LOCAL_IDS, MessageType, PacketHead,
                      ProtocolParser, Runtime, VERSION)
from shared_transport import (SharedPayloadArena, SharedPayloadError, SharedSelection,
                              pack_shared_receipt, prepare_shared_request,
                              resolve_shared_request, unpack_shared_receipt,
                              validate_reference)

MAX_FRAME_BYTES = 64 * 1024 * 1024
_FRAME_HEADER = struct.Struct("!i")
_RESPONSE_ID = struct.Struct("!Q")
_BINARY_MARKER = b"\x00THDV"
_BINARY_LENGTH = struct.Struct("!I")


@dataclass(frozen=True)
class BinaryReply:
    text: str
    binary: bytes


def _pack_binary_message(wire: bytes, binary: bytes) -> bytes:
    if not isinstance(binary, bytes) or not binary:
        raise ValueError("二进制决策载荷必须是非空 bytes")
    return _BINARY_MARKER + _BINARY_LENGTH.pack(len(wire)) + wire + binary


def _unpack_binary_message(raw: bytes) -> tuple[bytes, bytes | None]:
    if not raw.startswith(_BINARY_MARKER):
        return raw, None
    prefix = len(_BINARY_MARKER) + _BINARY_LENGTH.size
    if len(raw) < prefix:
        raise ValueError("二进制消息头不完整")
    wire_length = _BINARY_LENGTH.unpack(raw[len(_BINARY_MARKER):prefix])[0]
    if not wire_length or len(raw) <= prefix + wire_length:
        raise ValueError("二进制消息长度无效")
    return raw[prefix:prefix + wire_length], raw[prefix + wire_length:]


class IpcRuntimeState(str, Enum):
    READY = "READY"
    CASE_ACTIVE = "CASE_ACTIVE"
    FAILED = "FAILED"
    CLOSED = "CLOSED"


def _wait_for(fd: int, event: int, deadline: float | None) -> None:
    remaining = None if deadline is None else max(0.0, deadline - monotonic())
    if remaining == 0:
        raise TimeoutError("Pipe 传输超时")
    with selectors.DefaultSelector() as selector:
        selector.register(fd, event)
        if not selector.select(remaining):
            raise TimeoutError("Pipe 传输超时")


def _send_frame(connection: Connection, payload: bytes, deadline: float) -> None:
    """Write a multiprocessing-compatible frame without an unbounded send()."""
    if len(payload) > MAX_FRAME_BYTES:
        raise ValueError("IPC 报文超过大小上限")
    fd = connection.fileno()
    os.set_blocking(fd, False)
    frame = memoryview(_FRAME_HEADER.pack(len(payload)) + payload)
    while frame:
        _wait_for(fd, selectors.EVENT_WRITE, deadline)
        try:
            frame = frame[os.write(fd, frame):]
        except BlockingIOError:
            continue


def _read_exact(connection: Connection, count: int, deadline: float | None) -> bytes:
    fd = connection.fileno()
    os.set_blocking(fd, False)
    chunks = []
    while count:
        _wait_for(fd, selectors.EVENT_READ, deadline)
        try:
            chunk = os.read(fd, min(count, 65536))
        except BlockingIOError:
            continue
        if not chunk:
            raise EOFError("Pipe 已关闭")
        chunks.append(chunk)
        count -= len(chunk)
    return b"".join(chunks)


def _recv_frame(connection: Connection, deadline: float | None) -> bytes:
    size = _FRAME_HEADER.unpack(_read_exact(connection, _FRAME_HEADER.size, deadline))[0]
    if size < 0 or size > MAX_FRAME_BYTES:
        raise ValueError("IPC 帧长度无效")
    return _read_exact(connection, size, deadline)


def _send_shared_error(replies: Connection, name: str, response_id: int,
                       error: SharedPayloadError, timeout: float) -> None:
    response = AgentPacket(
        PacketHead(VERSION, response_id, error.trace_id, MessageType.ERROR.value,
                   name, error.message_id or None),
        {"error": str(error)},
    )
    wire = pack_shared_receipt(response.to_json().encode("utf-8"), 0, 0)
    _send_frame(replies, wire, monotonic() + timeout)


def _agent_worker(requests: Connection, replies: Connection, name: str,
                  capability: str, handler: Callable[[str], str],
                  timeout: float) -> None:
    parser = ProtocolParser()
    try:
        _send_frame(replies, b"READY", monotonic() + timeout)
        while True:
            try:
                frame = _recv_frame(requests, None)
            except EOFError:
                break
            if frame == b"STOP":
                break
            if len(frame) < _RESPONSE_ID.size:
                break
            response_id = _RESPONSE_ID.unpack(frame[:_RESPONSE_ID.size])[0]
            try:
                request_wire, copied_bytes, copy_ns, shared = resolve_shared_request(
                    frame[_RESPONSE_ID.size:], parser)
            except SharedPayloadError as exc:
                _send_shared_error(replies, name, response_id, exc, timeout)
                continue
            request = parser.parse(request_wire)
            if (request.head.message_type not in (MessageType.REQUEST.value, MessageType.RESPONSE.value)
                    or request.payload.get("capability") != capability):
                kind, payload = MessageType.ERROR, {"error": "Agent 收到无效请求"}
            else:
                try:
                    argument = (json.dumps(request.payload["data"], ensure_ascii=False)
                                if "data" in request.payload else request.payload["text"])
                    result = handler(argument)
                    if not isinstance(result, str):
                        raise TypeError("Agent Handler 必须返回字符串")
                    kind, payload = MessageType.RESPONSE, {"capability": capability, "text": result}
                except Exception as exc:
                    kind, payload = MessageType.ERROR, {"error": f"{type(exc).__name__}: {exc}"}
            response = AgentPacket(
                PacketHead(VERSION, response_id, request.head.trace_id, kind.value,
                           name, request.head.message_id), payload,
            )
            response_wire = response.to_json().encode("utf-8")
            if shared:
                response_wire = pack_shared_receipt(response_wire, copied_bytes, copy_ns)
            _send_frame(replies, response_wire, monotonic() + timeout)
    except (BrokenPipeError, EOFError, OSError, TimeoutError):
        pass
    finally:
        requests.close()
        replies.close()


def _managed_agent_worker(requests: Connection, replies: Connection, name: str,
                          capabilities: tuple[str, ...], factory: Callable[[], Any],
                          timeout: float) -> None:
    """Run a stateful Agent endpoint; management frames never enter TH-ACP."""
    parser = ProtocolParser()
    endpoint = None
    try:
        endpoint = factory()
        _send_frame(replies, b"READY", monotonic() + timeout)
        while True:
            try:
                frame = _recv_frame(requests, None)
            except EOFError:
                break
            if frame == b"STOP":
                break
            if len(frame) < _RESPONSE_ID.size:
                break
            response_id = _RESPONSE_ID.unpack(frame[:_RESPONSE_ID.size])[0]
            if response_id == 0:
                try:
                    command = json.loads(frame[_RESPONSE_ID.size:])
                    result = {"ok": True, "result": endpoint.control(command["op"], command.get("payload"))}
                except Exception as exc:
                    result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                _send_frame(replies, b"CONTROL" + json.dumps(
                    result, ensure_ascii=False).encode("utf-8"), monotonic() + timeout)
                continue
            try:
                request_wire, copied_bytes, copy_ns, shared = resolve_shared_request(
                    frame[_RESPONSE_ID.size:], parser)
                packet_wire, binary = _unpack_binary_message(request_wire)
            except SharedPayloadError as exc:
                _send_shared_error(replies, name, response_id, exc, timeout)
                continue
            except ValueError as exc:
                # The transport envelope is corrupt; the parent will turn the
                # missing correlated reply into a transport error and retire it.
                raise OSError(str(exc)) from exc
            decode_started_ns = perf_counter_ns()
            packet = parser.parse(packet_wire)
            if hasattr(endpoint, "transport_codec_stats"):
                endpoint.transport_codec_stats["request_decode_ns"] += perf_counter_ns() - decode_started_ns
            capability = packet.payload.get("capability")
            binary_response = None
            if (packet.head.message_type not in (MessageType.REQUEST.value, MessageType.RESPONSE.value)
                    or capability not in capabilities):
                kind, payload = MessageType.ERROR, {"error": "Agent 收到无效报文"}
            else:
                try:
                    output = endpoint.handle(packet, binary)
                    if isinstance(output, BinaryReply):
                        if capability not in ("doctor.recommend", "patient.request_history"):
                            raise TypeError("该 capability 不允许二进制响应")
                        binary_response = output.binary
                        output = output.text
                    else:
                        binary_response = None
                    if not isinstance(output, str):
                        raise TypeError("Agent Handler 必须返回字符串")
                    structured = None
                    if getattr(endpoint, "structured_data", False):
                        try:
                            structured = json.loads(output)
                        except (ValueError, TypeError):
                            pass
                    kind, payload = MessageType.RESPONSE, (
                        {"capability": capability, "data": structured}
                        if getattr(endpoint, "structured_data", False) and isinstance(structured, dict)
                        else {"capability": capability, "text": output})
                except Exception as exc:
                    kind, payload = MessageType.ERROR, {"error": f"{type(exc).__name__}: {exc}"}
                    binary_response = None
            response = AgentPacket(
                PacketHead(VERSION, response_id, packet.head.trace_id, kind.value,
                           name, packet.head.message_id), payload,
            )
            encode_started_ns = perf_counter_ns()
            response_wire = response.to_json().encode("utf-8")
            if hasattr(endpoint, "transport_codec_stats"):
                endpoint.transport_codec_stats["response_encode_ns"] += perf_counter_ns() - encode_started_ns
            if binary_response is not None:
                response_wire = _pack_binary_message(response_wire, binary_response)
            if shared:
                response_wire = pack_shared_receipt(response_wire, copied_bytes, copy_ns)
            _send_frame(replies, response_wire, monotonic() + timeout)
    except (BrokenPipeError, EOFError, OSError, TimeoutError):
        pass
    finally:
        if endpoint is not None and hasattr(endpoint, "close"):
            endpoint.close()
        requests.close()
        replies.close()


@dataclass
class _AgentProxy:
    name: str
    runtime: Runtime | None = None

    def make_packet(self, message_type: MessageType, *, payload: dict[str, Any]) -> AgentPacket:
        return AgentPacket.create(trace_id=0, message_type=message_type,
                                  source=self.name, payload=payload)


class MultiprocessRuntime(Runtime):
    """A synchronous Runtime with opt-in, one-process-per-Agent IPC handlers.

    One request is in flight per Runtime. The parent allocates all message IDs,
    including response IDs passed beside the serialized packet over the Pipe.
    A timeout or broken connection retires that Agent process.
    """

    def __init__(self, trace: Any, *, timeout: float = 2.0,
                 startup_timeout: float = 5.0,
                 binary_required: bool = True,
                 shared_medicines: bool = False) -> None:
        if timeout <= 0 or startup_timeout <= 0:
            raise ValueError("timeout 和 startup_timeout 必须大于 0")
        if type(binary_required) is not bool or type(shared_medicines) is not bool:
            raise ValueError("二进制决策状态开关必须为布尔值")
        super().__init__(trace)
        self.timeout = timeout
        self.startup_timeout = startup_timeout
        self.binary_required = binary_required
        self.shared_medicines = shared_medicines
        self._context = mp.get_context("spawn")
        self._workers: dict[str, tuple[mp.Process, Connection, Connection]] = {}
        self._process_capabilities: dict[str, str] = {}
        self._closed = False
        self.state = IpcRuntimeState.READY
        self.failure_reason: str | None = None
        self._outbound_binary: bytes | None = None
        self.binary_frame_stats: list[dict[str, Any]] = []
        self._outbound_shared: SharedSelection | None = None
        self._shared_arena = SharedPayloadArena()
        self._forwarded_segments: dict[str, tuple[shared_memory.SharedMemory, dict[str, Any]]] = {}
        self._medicine_reserved_name: str | None = None
        self._forwarded_write_bytes = 0
        self._forwarded_read_bytes = 0
        self.shared_memory_case_stats: list[dict[str, Any]] = []
        self.transport_frame_stats: list[dict[str, Any]] = []
        self.protocol_codec_stats: list[dict[str, int | str]] = []

    @property
    def shared_memory_active_bytes(self) -> int:
        return self._shared_arena.active_bytes + sum(
            ref["size"] for _segment, ref in self._forwarded_segments.values())

    def reserve_medicine_segment(self) -> str:
        """Reserve a name so a pre-reply worker crash still permits cleanup."""
        if (not self.shared_medicines or self.state != IpcRuntimeState.CASE_ACTIVE
                or self._medicine_reserved_name is not None):
            raise RuntimeError("当前病例无法预留药物共享段")
        self._medicine_reserved_name = "thipc_" + secrets.token_hex(16)
        return self._medicine_reserved_name

    def _register_forwarded_medicines(self, response: AgentPacket) -> int:
        body = (response.payload["data"] if "data" in response.payload
                else json.loads(response.payload["text"]))
        if (not isinstance(body, dict) or body.get("medicine_introductions", object()) is not None
                or type(body.get("case_id")) is not int):
            raise ValueError("Retrieval 共享药物说明回复格式无效")
        ref = body.get("shared_medicine_ref")
        validate_reference(ref, response.head.trace_id)
        if ref["name"] != self._medicine_reserved_name:
            raise ValueError("Retrieval 共享段名称与当前病例预留值不匹配")
        if ref["name"] in self._forwarded_segments:
            raise ValueError("Retrieval 共享引用重复")
        segment = shared_memory.SharedMemory(name=ref["name"], create=False)
        try:
            if segment.size != ref["size"]:
                raise ValueError("Retrieval 共享内存长度不匹配")
            self._forwarded_segments[ref["name"]] = (segment, dict(ref))
        except Exception:
            segment.close()
            raise
        self._forwarded_write_bytes += ref["size"]
        return ref["size"]

    def _check_forwarded_medicines(self, packet: AgentPacket) -> int:
        body = (packet.payload["data"] if "data" in packet.payload
                else json.loads(packet.payload["text"]))
        if (not isinstance(body, dict) or body.get("medicine_introductions", object()) is not None):
            raise ValueError("Expert 共享药物说明请求格式无效")
        ref = body.get("shared_medicine_ref")
        validate_reference(ref, packet.head.trace_id)
        registered = self._forwarded_segments.get(ref["name"])
        if registered is None or registered[1] != ref:
            raise ValueError("Expert 共享药物说明引用未登记或不匹配")
        return ref["size"]

    def _release_forwarded(self) -> None:
        released_names = set()
        for segment, _ref in self._forwarded_segments.values():
            released_names.add(segment.name)
            try:
                segment.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                self._shared_arena.cleanup_errors.append(f"{segment.name} unlink: {exc}")
            finally:
                try:
                    segment.close()
                except OSError as exc:
                    self._shared_arena.cleanup_errors.append(f"{segment.name} close: {exc}")
        self._forwarded_segments.clear()
        if (self._medicine_reserved_name is not None
                and self._medicine_reserved_name not in released_names):
            try:
                segment = shared_memory.SharedMemory(
                    name=self._medicine_reserved_name, create=False)
            except FileNotFoundError:
                pass
            except OSError as exc:
                self._shared_arena.cleanup_errors.append(
                    f"{self._medicine_reserved_name} attach: {exc}")
            else:
                try:
                    segment.unlink()
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    self._shared_arena.cleanup_errors.append(
                        f"{self._medicine_reserved_name} unlink: {exc}")
                finally:
                    try:
                        segment.close()
                    except OSError as exc:
                        self._shared_arena.cleanup_errors.append(
                            f"{self._medicine_reserved_name} close: {exc}")
        self._medicine_reserved_name = None

    def deliver_shared(self, raw: AgentPacket, *, fields: tuple[str, ...] | None = None,
                       min_bytes: int = 256) -> Delivery:
        """Select a whole TH-ACP packet or JSON text fields for shared memory.

        ``fields=None`` selects the entire serialized packet. JSON Pointer paths
        select string fields inside ``payload.text``. Values below ``min_bytes``
        remain on the ordinary Pipe path.
        """
        if not isinstance(raw, AgentPacket):
            raise TypeError("共享内存投递需要 AgentPacket")
        if type(min_bytes) is not int or not 0 < min_bytes <= 64 * 1024 * 1024:
            raise ValueError("共享内存阈值无效")
        if fields is not None and (not isinstance(fields, tuple)
                                   or not fields or any(not isinstance(path, str) for path in fields)):
            raise ValueError("共享字段必须是非空 JSON Pointer 元组")
        if self._outbound_shared is not None or self._outbound_binary is not None:
            raise RuntimeError("已有待发送的特殊载荷")
        self._outbound_shared = SharedSelection(fields, min_bytes)
        try:
            return self.deliver(raw)
        finally:
            self._outbound_shared = None

    def deliver_binary(self, raw: AgentPacket, binary: bytes) -> Delivery:
        """Send one TH-ACP packet with an opaque binary Pipe sidecar."""
        if raw.payload.get("capability") not in ("expert.evaluate", "retrieval.history"):
            raise ValueError("二进制载荷仅用于药物决策或查询向量")
        if not isinstance(binary, bytes) or not binary:
            raise ValueError("二进制载荷必须是非空 bytes")
        if self._outbound_binary is not None or self._outbound_shared is not None:
            raise RuntimeError("已有待发送的二进制载荷")
        self._outbound_binary = binary
        try:
            return self.deliver(raw)
        finally:
            self._outbound_binary = None

    def start_case(self) -> int:
        if self.state != IpcRuntimeState.READY:
            raise RuntimeError(f"IPC Runtime 当前状态为 {self.state.value}，无法开始病例")
        trace_id = super().start_case()
        self.state = IpcRuntimeState.CASE_ACTIVE
        return trace_id

    def finish_case(self) -> None:
        trace_id = self.trace_id
        super().finish_case()
        self._release_shared_case(trace_id)
        if self.state == IpcRuntimeState.CASE_ACTIVE:
            self.state = IpcRuntimeState.READY

    def _release_shared_case(self, trace_id: int | None) -> None:
        self._release_forwarded()
        if trace_id is None:
            self._shared_arena.release()
            self._forwarded_write_bytes = self._forwarded_read_bytes = 0
            return
        if (self.shared_memory_case_stats
                and self.shared_memory_case_stats[-1]["trace_id"] == trace_id):
            return
        statistics = self._shared_arena.release()
        self.shared_memory_case_stats.append({
            "trace_id": trace_id, **statistics,
            "forwarded_write_bytes": self._forwarded_write_bytes,
            "forwarded_read_bytes": self._forwarded_read_bytes,
            "cleanup_errors": list(self._shared_arena.cleanup_errors),
        })
        self._forwarded_write_bytes = self._forwarded_read_bytes = 0
        self._shared_arena.cleanup_errors.clear()

    def register_process_agent(self, name: str, capability: str,
                               handler: Callable[[str], str]) -> None:
        """Register a picklable, text-to-text handler in its own process."""
        self._register_worker(name, (capability,), _agent_worker,
                              (name, capability, handler, self.timeout))

    def register_managed_agent(self, name: str, capabilities: tuple[str, ...],
                               factory: Callable[[], Any]) -> None:
        """Register a stateful endpoint with several capabilities in one child."""
        if not capabilities or len(capabilities) != len(set(capabilities)):
            raise ValueError("Agent capabilities 必须非空且不重复")
        self._register_worker(name, capabilities, _managed_agent_worker,
                              (name, capabilities, factory, self.timeout))

    def _register_worker(self, name: str, capabilities: tuple[str, ...],
                         target: Callable[..., None], args: tuple[Any, ...]) -> None:
        if self.state != IpcRuntimeState.READY:
            raise RuntimeError(f"IPC Runtime 当前状态为 {self.state.value}，无法注册 Agent")
        if (name in self.agents or name in self._workers
                or any(self.capabilities.resolve(capability) for capability in capabilities)):
            raise ValueError("Agent 或 capability 重复注册")
        child_requests, parent_requests = self._context.Pipe(duplex=False)
        parent_replies, child_replies = self._context.Pipe(duplex=False)
        process = self._context.Process(target=target,
                                        args=(child_requests, child_replies, *args),
                                        name=f"TH-ACP-{name}")
        try:
            process.start()
            child_requests.close()
            child_replies.close()
            try:
                ready = _recv_frame(parent_replies, monotonic() + self.startup_timeout)
            except TimeoutError as exc:
                raise RuntimeError("Agent 子进程就绪超时") from exc
            except (EOFError, OSError, ValueError) as exc:
                raise RuntimeError("Agent 子进程启动失败") from exc
            if ready != b"READY" or not process.is_alive():
                raise RuntimeError("Agent 子进程未就绪")
            self.register_agent(_AgentProxy(name), {
                capability: lambda _text: None for capability in capabilities
            })
            self._workers[name] = (process, parent_requests, parent_replies)
            self._process_capabilities.update({capability: name for capability in capabilities})
        except Exception:
            child_requests.close()
            child_replies.close()
            parent_requests.close()
            parent_replies.close()
            if process.pid is not None:
                process.terminate()
                process.join()
            raise

    def _dispatch(self, packet: AgentPacket) -> Delivery:
        if self.state in (IpcRuntimeState.FAILED, IpcRuntimeState.CLOSED):
            return self._error(packet, self.failure_reason or "IPC Runtime 已关闭")
        capability = packet.payload.get("capability")
        if (packet.head.message_type not in (MessageType.REQUEST.value, MessageType.RESPONSE.value)
                or capability not in self._process_capabilities):
            if self._outbound_shared is not None:
                return self._error(packet, "共享内存投递需要已注册的进程 Agent")
            return super()._dispatch(packet)
        if packet.head.trace_id != self.trace_id:
            return self._error(packet, "trace_id 与当前任务不匹配")
        if packet.head.source != "System" and packet.head.source not in self.agents:
            return self._error(packet, "未知发送方 Agent")
        name = self._process_capabilities[capability]
        worker = self._workers.get(name)
        if worker is None:
            reason = f"Agent {name} 子进程不可用"
            self._fail_transport(reason)
            return self._error(packet, reason)
        process, requests, replies = worker
        if not process.is_alive():
            reason = f"Agent {name} 子进程已退出"
            self._fail_transport(reason)
            return self._error(packet, reason)

        # Reserve the response ID in the parent before sending; no child-local
        # counter can collide with an ID from another Agent on the same trace.
        response_id = LOCAL_IDS.next_message(packet.head.trace_id)
        if response_id > (1 << 64) - 1:
            return self._error(packet, "message_id 超过 IPC 传输上限")
        content = packet.payload.get("data", packet.payload.get("text"))
        if not isinstance(content, (str, dict)):
            return self._error(packet, "IPC Agent 当前只支持文本或 JSON 对象报文")
        forwarded_read_bytes = 0
        if self.shared_medicines and capability == "expert.review":
            try:
                forwarded_read_bytes = self._check_forwarded_medicines(packet)
            except (ValueError, OSError) as exc:
                return self._error(packet, f"共享药物说明引用无效：{exc}")
        self.trace.message(packet.head.source, name, content if isinstance(content, str)
                           else json.dumps(content, ensure_ascii=False))
        deadline = monotonic() + self.timeout
        encode_started_ns = perf_counter_ns()
        packet_wire = packet.to_json().encode("utf-8")
        encode_ns = perf_counter_ns() - encode_started_ns
        binary = self._outbound_binary if capability in ("expert.evaluate", "retrieval.history") else None
        if self.binary_required and capability == "expert.evaluate" and binary is None:
            return self._error(packet, "Expert 推荐结果缺少二进制载荷")
        if self._outbound_shared is not None:
            try:
                request_payload, referenced_bytes, written_bytes, write_ns = prepare_shared_request(
                    packet, self._outbound_shared, self._shared_arena)
            except (ValueError, OSError) as exc:
                return self._error(packet, f"共享载荷准备失败：{exc}")
        else:
            request_payload = packet_wire
            referenced_bytes = written_bytes = write_ns = 0
        shared = referenced_bytes > 0
        if monotonic() >= deadline:
            return self._error(packet, "共享载荷准备超时")
        started_ns = perf_counter_ns()
        try:
            wire = _RESPONSE_ID.pack(response_id) + (
                _pack_binary_message(request_payload, binary)
                if binary is not None else request_payload)
            _send_frame(requests, wire, deadline)
            if binary is not None:
                self.binary_frame_stats.append({
                    "trace_id": packet.head.trace_id, "capability": capability,
                    "direction": "request", "frame_bytes": _FRAME_HEADER.size + len(wire),
                    "th_acp_bytes": len(packet_wire), "binary_bytes": len(binary),
                })
            raw_reply = _recv_frame(replies, deadline)
            reply_payload, read_bytes, read_ns, receipt = unpack_shared_receipt(raw_reply)
            if receipt != shared:
                raise ValueError("共享载荷回执与请求不匹配")
            reply_wire, reply_binary = _unpack_binary_message(reply_payload)
            decode_started_ns = perf_counter_ns()
            response = self.parser.parse(reply_wire)
            decode_ns = perf_counter_ns() - decode_started_ns
            if (shared and response.head.message_type == MessageType.RESPONSE.value
                    and read_bytes != referenced_bytes):
                raise ValueError("共享载荷读取字节数不匹配")
        except TimeoutError:
            reason = f"Agent {name} IPC 传输或响应超时（capability={capability}，任务上限={self.timeout:g}秒，含模型等待及内容重试）"
            self._fail_transport(reason)
            return self._error(packet, reason)
        except (BrokenPipeError, EOFError, OSError, ValueError, struct.error) as exc:
            reason = f"Agent {name} 子进程连接中断：{exc}"
            self._fail_transport(reason)
            return self._error(packet, reason)

        forwarded_write_bytes = 0
        if (self.shared_medicines and capability == "retrieval.medicines"
                and response.head.message_type == MessageType.RESPONSE.value):
            try:
                forwarded_write_bytes = self._register_forwarded_medicines(response)
            except (ValueError, OSError) as exc:
                reason = f"Retrieval 共享药物说明引用无效：{exc}"
                self._fail_transport(reason)
                return self._error(packet, reason)
        if (self.shared_medicines and capability == "expert.review"
                and response.head.message_type == MessageType.RESPONSE.value):
            self._forwarded_read_bytes += forwarded_read_bytes

        self.transport_frame_stats.append({
            "trace_id": packet.head.trace_id, "capability": capability,
            "mode": "shared" if shared else "binary" if binary is not None else "pipe",
            "request_frame_bytes": _FRAME_HEADER.size + len(wire),
            "reply_frame_bytes": _FRAME_HEADER.size + len(raw_reply),
            "logical_request_bytes": len(packet_wire),
            "shared_reference_bytes": referenced_bytes,
            "shared_new_bytes": written_bytes, "shared_write_ns": write_ns,
            "shared_read_bytes": read_bytes, "shared_read_ns": read_ns,
            "forwarded_shared_write_bytes": forwarded_write_bytes,
            "forwarded_shared_read_bytes": (forwarded_read_bytes if
                                             response.head.message_type == MessageType.RESPONSE.value else 0),
            "round_trip_ns": perf_counter_ns() - started_ns,
        })
        self.protocol_codec_stats.append({
            "trace_id": packet.head.trace_id, "capability": capability,
            "request_encode_ns": encode_ns, "response_decode_ns": decode_ns,
        })

        head = response.head
        if (head.message_type not in (MessageType.RESPONSE.value, MessageType.ERROR.value)
                or head.message_id != response_id
                or head.trace_id != packet.head.trace_id
                or head.correlation_id != packet.head.message_id
                or head.source != name
                or (head.message_type == MessageType.RESPONSE.value
                    and response.payload["capability"] != capability)
                or (reply_binary is not None and (capability not in ("doctor.recommend", "patient.request_history")
                    or head.message_type != MessageType.RESPONSE.value))
                or (self.binary_required and capability == "doctor.recommend" and head.message_type == MessageType.RESPONSE.value
                    and reply_binary is None)):
            reason = f"Agent {name} 响应关联校验失败"
            self._fail_transport(reason)
            return self._error(packet, reason)
        response_index = self.trace.packet(response)
        self.trace.packet_costs[response_index] = replace(
            self.trace.packet_costs[response_index], target="Runtime")
        self.trace.finish_delivery(response_index, 0, 0)
        if reply_binary is not None:
            self.binary_frame_stats.append({
                "trace_id": packet.head.trace_id, "capability": capability,
                "direction": "response", "frame_bytes": _FRAME_HEADER.size + len(raw_reply),
                "th_acp_bytes": len(reply_wire), "binary_bytes": len(reply_binary),
            })
            reply_text = (response.payload["text"] if "text" in response.payload else
                          json.dumps(response.payload["data"], ensure_ascii=False))
            return Delivery(response, BinaryReply(reply_text, reply_binary))
        return Delivery(response, response.payload.get("text") if "text" in response.payload
                        else (json.dumps(response.payload["data"], ensure_ascii=False)
                              if "data" in response.payload else None))

    def control_agent(self, name: str, op: str, payload: Any = None) -> Any:
        """Send a bounded lifecycle command to a managed child process."""
        if self.state in (IpcRuntimeState.FAILED, IpcRuntimeState.CLOSED):
            raise RuntimeError(self.failure_reason or "IPC Runtime 已关闭")
        worker = self._workers.get(name)
        if worker is None:
            reason = f"Agent {name} 子进程不可用"
            self._fail_transport(reason)
            raise RuntimeError(reason)
        process, requests, replies = worker
        if not process.is_alive():
            reason = f"Agent {name} 子进程已退出"
            self._fail_transport(reason)
            raise RuntimeError(reason)
        deadline = monotonic() + self.timeout
        frame = _RESPONSE_ID.pack(0) + json.dumps(
            {"op": op, "payload": payload}, ensure_ascii=False).encode("utf-8")
        try:
            _send_frame(requests, frame, deadline)
            reply = _recv_frame(replies, deadline)
            if not reply.startswith(b"CONTROL"):
                raise ValueError("Agent 管理响应格式错误")
            result = json.loads(reply[len(b"CONTROL"):])
        except (BrokenPipeError, EOFError, OSError, TimeoutError, ValueError) as exc:
            reason = f"Agent {name} 管理通信失败：{exc}"
            self._fail_transport(reason)
            raise RuntimeError(reason) from exc
        if not result.get("ok"):
            raise RuntimeError(f"Agent {name} 管理操作失败：{result.get('error')}")
        return result["result"]

    def _fail_transport(self, reason: str) -> None:
        """A lost reply leaves side effects uncertain; retire every Agent."""
        if self.state in (IpcRuntimeState.FAILED, IpcRuntimeState.CLOSED):
            return
        self.failure_reason = reason
        self.state = IpcRuntimeState.FAILED
        for name in list(self._workers):
            self._retire(name)
        self._release_shared_case(self.trace_id)

    def fail(self, reason: str) -> None:
        """Retire all workers when the parent cannot establish a safe case state."""
        self._fail_transport(reason)

    def _retire(self, name: str) -> None:
        worker = self._workers.pop(name, None)
        if worker is None:
            return
        process, requests, replies = worker
        for connection in (requests, replies):
            try:
                connection.close()
            except OSError:
                pass
        try:
            if process.is_alive():
                process.terminate()
            process.join(timeout=1)
            if process.is_alive():
                process.kill()
                process.join()
        except OSError:
            # Another thread or the OS may have reaped an exiting child.
            process.join(timeout=1)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for name, (process, requests, _replies) in list(self._workers.items()):
            try:
                if process.is_alive():
                    _send_frame(requests, b"STOP", monotonic() + min(self.timeout, 0.2))
                    process.join(timeout=1)
            except (BrokenPipeError, EOFError, OSError, TimeoutError):
                pass
            finally:
                self._retire(name)
        if self.trace_id is not None:
            self._release_shared_case(self.trace_id)
            super().finish_case()
        else:
            self._shared_arena.release()
            self._release_forwarded()
        self.state = IpcRuntimeState.CLOSED

    def __enter__(self) -> MultiprocessRuntime:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()
