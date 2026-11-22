"""Atomic writes for JSON row stores."""

from __future__ import annotations

import json
from pathlib import Path


def write_json_rows(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp')
    try:
        temporary.write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
