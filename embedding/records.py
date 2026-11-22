"""Read input texts without exposing dataset answer labels to the encoder."""
from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path

PREPROCESS_VERSION = 'nfc-lines-v1'


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError('文本必须是字符串')
    text = unicodedata.normalize('NFC', text).replace('\r\n', '\n').replace('\r', '\n')
    text = '\n'.join(line.strip() for line in text.split('\n') if line.strip())
    if not text:
        raise ValueError('文本不能为空')
    return text


@dataclass(frozen=True)
class TextRecord:
    record_id: str
    text: str
    kind: str
    source: str


def load_records(path: Path, kind: str) -> list[TextRecord]:
    """Read a complete source snapshot. Existing DialMed IDs are physical line numbers.

    Explicit record_id/memory_id/id is preferred; otherwise memory uses case_id.
    Only dialog/experience/reflection/text is encoded, never label or disease.
    """
    path = Path(path).resolve()
    raw = path.read_text(encoding='utf-8')
    if kind in ('dialog', 'text'):
        rows = [(i, json.loads(line)) for i, line in enumerate(raw.splitlines(), 1)
                if line.strip()]
    elif kind in ('experience', 'reflection'):
        value = json.loads(raw)
        if not isinstance(value, list):
            raise ValueError('经验/反思文件必须是 JSON 数组')
        rows = list(enumerate(value, 1))
    else:
        raise ValueError('kind 必须为 dialog、experience、reflection 或 text')
    records, seen = [], set()
    for line, row in rows:
        if not isinstance(row, dict):
            raise ValueError(f'{path}:{line} 必须是对象')
        identity = next((row[key] for key in ('record_id', 'memory_id', 'id') if key in row), None)
        if identity is None:
            identity = line if kind == 'dialog' else row.get('case_id')
        if type(identity) not in (str, int) or not str(identity).strip():
            raise ValueError(f'{path}:{line} 缺少有效记录 ID')
        record_id = str(identity)
        if record_id in seen:
            raise ValueError(f'重复记录 ID: {record_id}')
        seen.add(record_id)
        if kind == 'dialog':
            dialog = row.get('dialog')
            if not isinstance(dialog, list) or not dialog or any(not isinstance(x, str) for x in dialog):
                raise ValueError(f'{path}:{line} dialog 必须是非空字符串数组')
            text = clean_text('\n'.join(dialog))
        else:
            text = clean_text(row.get(kind))
        records.append(TextRecord(record_id, text, kind, str(path)))
    return records
