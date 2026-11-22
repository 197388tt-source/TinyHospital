"""Binary medicine decision vector with ordered indices for Expert handoff."""

from __future__ import annotations

import json
import struct
from hashlib import blake2b
from pathlib import Path

from tool.medicine_store import MEDICINE_ROOT

_HEADER = struct.Struct("!4sBHB8s")
_MAGIC = b"THDV"
_VERSION = 1


class DecisionVectorCodec:
    def __init__(self, catalog_path: Path = MEDICINE_ROOT / "medicine.json") -> None:
        mapping = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
        if (not isinstance(mapping, dict) or not 0 < len(mapping) <= 256
                or any(not isinstance(name, str) or not name.strip()
                       or type(index) is not int for name, index in mapping.items())
                or sorted(mapping.values()) != list(range(len(mapping)))):
            raise ValueError("药物编号表必须包含连续且唯一的 0–255 编号")
        self.names = tuple(name for name, _index in sorted(mapping.items(), key=lambda item: item[1]))
        self.indices = {name: index for index, name in enumerate(self.names)}
        self.vector_bytes = (len(self.names) + 7) // 8
        catalog = json.dumps(self.names, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.fingerprint = blake2b(catalog, digest_size=8).digest()

    def encode(self, recommended_drugs: list[str]) -> bytes:
        if (not isinstance(recommended_drugs, list) or not recommended_drugs
                or len(recommended_drugs) > 255
                or any(not isinstance(name, str) or name not in self.indices
                       for name in recommended_drugs)
                or len(recommended_drugs) != len(set(recommended_drugs))):
            raise ValueError("推荐药物必须是不重复的编号表药名")
        vector = bytearray(self.vector_bytes)
        order = bytearray()
        for name in recommended_drugs:
            index = self.indices[name]
            vector[index // 8] |= 1 << (index % 8)
            order.append(index)
        return (_HEADER.pack(_MAGIC, _VERSION, len(self.names), len(order), self.fingerprint)
                + vector + order)

    def decode(self, raw: bytes) -> list[str]:
        if not isinstance(raw, bytes) or len(raw) < _HEADER.size:
            raise ValueError("二进制决策状态格式无效")
        magic, version, count, selected, fingerprint = _HEADER.unpack(raw[:_HEADER.size])
        if (magic != _MAGIC or version != _VERSION or count != len(self.names)
                or fingerprint != self.fingerprint or selected == 0
                or len(raw) != _HEADER.size + self.vector_bytes + selected):
            raise ValueError("二进制决策状态头部、目录或长度无效")
        vector = raw[_HEADER.size:_HEADER.size + self.vector_bytes]
        order = raw[_HEADER.size + self.vector_bytes:]
        if (any(index >= count for index in order)
                or len(set(order)) != selected
                or sum(byte.bit_count() for byte in vector) != selected
                or any(not (vector[index // 8] & (1 << (index % 8))) for index in order)):
            raise ValueError("二进制决策向量与推荐顺序不一致")
        return [self.names[index] for index in order]
