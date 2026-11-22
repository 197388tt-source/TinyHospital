"""Reusable, case-scoped shared-memory transport for TH-ACP Pipe messages."""

from __future__ import annotations

import json
import re
import secrets
import struct
import sys
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from multiprocessing import shared_memory
from time import perf_counter_ns
from typing import Any

from protocol import AgentPacket, MessageType, ProtocolParser

MAX_SHARED_BYTES = 64 * 1024 * 1024
MAX_CASE_SHARED_BYTES = 128 * 1024 * 1024
_MARKER = b"\x00THSM"
_RECEIPT_MARKER = b"\x00THSR"
_LENGTH = struct.Struct("!I")
_RECEIPT = struct.Struct("!QQ")
_NAME = re.compile(r"thipc_[0-9a-f]{32}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class SharedPayloadError(ValueError):
    def __init__(self, message: str, trace_id: int = 0, message_id: int = 0) -> None:
        super().__init__(message)
        self.trace_id = trace_id
        self.message_id = message_id


@dataclass(frozen=True)
class SharedSelection:
    fields: tuple[str, ...] | None
    min_bytes: int


def _pointer(root: Any, path: str) -> tuple[Any, str | int]:
    """Return the parent and last key for a JSON Pointer."""
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError("共享字段必须使用 JSON Pointer 路径")
    parts = path[1:].split("/")
    keys = []
    for part in parts:
        if re.search(r"~(?![01])", part):
            raise ValueError("共享字段路径转义无效")
        keys.append(part.replace("~1", "/").replace("~0", "~"))
    node = root
    for key in keys[:-1]:
        node = _child(node, key)
    final = keys[-1]
    if isinstance(node, list):
        if not final.isdigit() or (len(final) > 1 and final[0] == "0"):
            raise ValueError("共享字段列表索引无效")
        return node, int(final)
    if isinstance(node, dict):
        return node, final
    raise ValueError("共享字段路径不存在")


def _child(node: Any, key: str) -> Any:
    if isinstance(node, dict):
        if key not in node:
            raise ValueError("共享字段路径不存在")
        return node[key]
    if isinstance(node, list) and key.isdigit() and (len(key) == 1 or key[0] != "0"):
        index = int(key)
        if index < len(node):
            return node[index]
    raise ValueError("共享字段路径不存在")


def _get(root: Any, path: str) -> tuple[Any, str | int, Any]:
    parent, key = _pointer(root, path)
    if isinstance(parent, dict) and key in parent:
        return parent, key, parent[key]
    if isinstance(parent, list) and isinstance(key, int) and key < len(parent):
        return parent, key, parent[key]
    raise ValueError("共享字段路径不存在")


class SharedPayloadArena:
    """Deduplicate selected bytes for one case and unlink them on every exit."""

    def __init__(self) -> None:
        self._segments: dict[tuple[int, str], tuple[shared_memory.SharedMemory, dict[str, Any]]] = {}
        self.active_bytes = 0
        self.peak_bytes = 0
        self.allocations = 0
        self.write_bytes = 0
        self.write_ns = 0
        self.cleanup_errors: list[str] = []

    @property
    def active_names(self) -> tuple[str, ...]:
        return tuple(item[0].name for item in self._segments.values())

    def publish(self, data: bytes, trace_id: int, *, name: str | None = None
                ) -> tuple[dict[str, Any], int, int]:
        if (type(trace_id) is not int or trace_id <= 0 or not isinstance(data, bytes)
                or not 0 < len(data) <= MAX_SHARED_BYTES):
            raise ValueError("共享载荷类型、病例编号或大小无效")
        digest = sha256(data).hexdigest()
        key = (trace_id, digest)
        existing = self._segments.get(key)
        if existing is not None:
            return dict(existing[1]), 0, 0
        if self.active_bytes + len(data) > MAX_CASE_SHARED_BYTES:
            raise ValueError("当前病例共享内存超过上限")
        if name is None:
            name = "thipc_" + secrets.token_hex(16)
        elif not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError("共享内存段名无效")
        segment = shared_memory.SharedMemory(name=name, create=True, size=len(data))
        try:
            started = perf_counter_ns()
            segment.buf[:len(data)] = data
            copied_ns = perf_counter_ns() - started
        except Exception:
            segment.close()
            segment.unlink()
            raise
        ref = {"name": name, "size": len(data), "sha256": digest, "trace_id": trace_id}
        self._segments[key] = (segment, ref)
        self.active_bytes += len(data)
        self.peak_bytes = max(self.peak_bytes, self.active_bytes)
        self.allocations += 1
        self.write_bytes += len(data)
        self.write_ns += copied_ns
        return dict(ref), copied_ns, len(data)

    def release(self) -> dict[str, int]:
        statistics = {"peak_bytes": self.peak_bytes, "allocations": self.allocations,
                      "write_bytes": self.write_bytes, "write_ns": self.write_ns}
        for segment, _ref in self._segments.values():
            try:
                segment.close()
            except OSError as exc:
                self.cleanup_errors.append(f"{segment.name} close: {exc}")
            try:
                segment.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                self.cleanup_errors.append(f"{segment.name} unlink: {exc}")
        self._segments.clear()
        self.active_bytes = 0
        self.peak_bytes = 0
        self.allocations = 0
        self.write_bytes = 0
        self.write_ns = 0
        return statistics


def validate_reference(ref: Any, trace_id: int) -> None:
    if (not isinstance(ref, dict) or set(ref) != {"name", "size", "sha256", "trace_id"}
            or type(ref["trace_id"]) is not int or ref["trace_id"] != trace_id
            or type(ref["size"]) is not int or not 0 < ref["size"] <= MAX_SHARED_BYTES
            or not isinstance(ref["name"], str) or not _NAME.fullmatch(ref["name"])
            or not isinstance(ref["sha256"], str) or not _DIGEST.fullmatch(ref["sha256"])):
        raise ValueError("共享内存引用无效或病例不匹配")


def attach_shared_memory(name: str) -> shared_memory.SharedMemory:
    """Attach to a session-owned segment on Python 3.10+.

    Older Python uses the common multiprocessing resource tracker; the
    session owner still unlinks the segment. Do not manually unregister it,
    which would also remove the owner's registration from that tracker.
    """
    options = {"track": False} if sys.version_info >= (3, 13) else {}
    return shared_memory.SharedMemory(name=name, create=False, **options)


def read_reference(ref: Any, trace_id: int) -> tuple[bytes, int]:
    validate_reference(ref, trace_id)
    try:
        segment = attach_shared_memory(ref["name"])
    except (FileNotFoundError, OSError) as exc:
        raise ValueError("共享内存引用已失效") from exc
    try:
        if segment.size != ref["size"]:
            raise ValueError("共享内存长度不匹配")
        started = perf_counter_ns()
        data = bytes(segment.buf[:ref["size"]])
        copied_ns = perf_counter_ns() - started
        if sha256(data).hexdigest() != ref["sha256"]:
            raise ValueError("共享内存内容校验失败")
        return data, copied_ns
    finally:
        segment.close()


def prepare_shared_request(packet: AgentPacket, selection: SharedSelection,
                           arena: SharedPayloadArena) -> tuple[bytes, int, int, int]:
    """Return Pipe payload, referenced bytes, newly copied bytes and copy time."""
    plain = packet.to_json().encode("utf-8")
    identity = {"trace_id": packet.head.trace_id, "message_id": packet.head.message_id}
    if selection.fields is None:
        if len(plain) < selection.min_bytes:
            return plain, 0, 0, 0
        ref, copied_ns, copied_bytes = arena.publish(plain, packet.head.trace_id)
        meta = {**identity, "mode": "packet", "ref": ref}
        return _envelope(meta, b""), len(plain), copied_bytes, copied_ns
    if not selection.fields:
        raise ValueError("共享字段列表不能为空")
    if len(selection.fields) != len(set(selection.fields)):
        raise ValueError("共享字段路径不可重复")
    structured = "data" in packet.payload
    body = deepcopy(packet.payload["data"]) if structured else json.loads(packet.payload["text"])
    fields = []
    copied_bytes = copied_ns = referenced_bytes = 0
    for path in selection.fields:
        parent, key, value = _get(body, path)
        if not isinstance(value, str):
            raise ValueError("共享字段必须是字符串")
        data = value.encode("utf-8")
        if len(data) < selection.min_bytes:
            continue
        ref, write_ns, written = arena.publish(data, packet.head.trace_id)
        parent[key] = None
        fields.append({"path": path, "ref": ref})
        referenced_bytes += len(data)
        copied_bytes += written
        copied_ns += write_ns
    if not fields:
        return plain, 0, 0, 0
    light_packet = AgentPacket(packet.head, {**packet.payload,
        "data": body} if structured else {**packet.payload,
        "text": json.dumps(body, ensure_ascii=False)})
    meta = {**identity, "mode": "fields", "fields": fields}
    return _envelope(meta, light_packet.to_json().encode("utf-8")), referenced_bytes, copied_bytes, copied_ns


def _envelope(meta: dict[str, Any], wire: bytes) -> bytes:
    metadata = json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return _MARKER + _LENGTH.pack(len(metadata)) + metadata + wire


def resolve_shared_request(raw: bytes, parser: ProtocolParser) -> tuple[bytes, int, int, bool]:
    if not raw.startswith(_MARKER):
        return raw, 0, 0, False
    trace_id = message_id = 0
    try:
        prefix = len(_MARKER) + _LENGTH.size
        if len(raw) < prefix:
            raise ValueError("共享引用消息头不完整")
        metadata_length = _LENGTH.unpack(raw[len(_MARKER):prefix])[0]
        if not 0 < metadata_length <= 64 * 1024 or len(raw) < prefix + metadata_length:
            raise ValueError("共享引用元数据长度无效")
        meta = json.loads(raw[prefix:prefix + metadata_length])
        if not isinstance(meta, dict):
            raise ValueError("共享引用元数据无效")
        trace_id, message_id = meta.get("trace_id"), meta.get("message_id")
        if (type(trace_id) is not int or trace_id <= 0
                or type(message_id) is not int or message_id <= 0):
            raise ValueError("共享引用报文编号无效")
        tail = raw[prefix + metadata_length:]
        if meta.get("mode") == "packet" and set(meta) == {"trace_id", "message_id", "mode", "ref"}:
            if tail:
                raise ValueError("整报文共享引用含多余数据")
            wire, copy_ns = read_reference(meta["ref"], trace_id)
            copied_bytes = len(wire)
        elif meta.get("mode") == "fields" and set(meta) == {"trace_id", "message_id", "mode", "fields"}:
            light = parser.parse(tail)
            if (light.head.message_type not in (MessageType.REQUEST.value, MessageType.RESPONSE.value)
                    or light.head.trace_id != trace_id or light.head.message_id != message_id):
                raise ValueError("共享字段报文关联无效")
            fields = meta["fields"]
            if not isinstance(fields, list) or not fields or len(fields) > 64:
                raise ValueError("共享字段引用列表无效")
            structured = "data" in light.payload
            body = light.payload["data"] if structured else json.loads(light.payload["text"])
            seen = set()
            copy_ns = copied_bytes = 0
            for item in fields:
                if not isinstance(item, dict) or set(item) != {"path", "ref"}:
                    raise ValueError("共享字段引用格式无效")
                path = item["path"]
                if path in seen:
                    raise ValueError("共享字段引用路径重复")
                seen.add(path)
                parent, key, placeholder = _get(body, path)
                if placeholder is not None:
                    raise ValueError("共享字段占位符无效")
                data, read_ns = read_reference(item["ref"], trace_id)
                parent[key] = data.decode("utf-8")
                copy_ns += read_ns
                copied_bytes += len(data)
            restored_payload = ({**light.payload, "data": body} if structured else
                                {**light.payload, "text": json.dumps(body, ensure_ascii=False)})
            wire = AgentPacket(light.head, restored_payload).to_json().encode("utf-8")
        else:
            raise ValueError("共享引用模式无效")
        packet = parser.parse(wire)
        if (packet.head.message_type not in (MessageType.REQUEST.value, MessageType.RESPONSE.value)
                or packet.head.trace_id != trace_id or packet.head.message_id != message_id):
            raise ValueError("共享引用与 TH-ACP 报文不匹配")
        return wire, copied_bytes, copy_ns, True
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
        raise SharedPayloadError(f"共享载荷解析失败：{exc}",
                                 trace_id if type(trace_id) is int and trace_id > 0 else 0,
                                 message_id if type(message_id) is int and message_id > 0 else 0) from exc


def pack_shared_receipt(wire: bytes, copied_bytes: int, copy_ns: int) -> bytes:
    return _RECEIPT_MARKER + _RECEIPT.pack(copied_bytes, copy_ns) + wire


def unpack_shared_receipt(raw: bytes) -> tuple[bytes, int, int, bool]:
    if not raw.startswith(_RECEIPT_MARKER):
        return raw, 0, 0, False
    prefix = len(_RECEIPT_MARKER) + _RECEIPT.size
    if len(raw) <= prefix:
        raise ValueError("共享载荷回执不完整")
    copied_bytes, copy_ns = _RECEIPT.unpack(raw[len(_RECEIPT_MARKER):prefix])
    return raw[prefix:], copied_bytes, copy_ns, True
