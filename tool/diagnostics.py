"""Append diagnostics; golden-experience review events include model replies, never API credentials."""
import json
from datetime import datetime, timezone
from pathlib import Path


def emit(path, event):
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {'timestamp': datetime.now(timezone.utc).isoformat(), **event}
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
