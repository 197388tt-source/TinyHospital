"""Append model token totals without generating case reports."""

from __future__ import annotations

import json
from pathlib import Path


def append_llm_usage(path: Path, case_id: int, calls: list[dict]) -> None:
    keys = ('prompt_tokens', 'completion_tokens', 'total_tokens')
    totals = {key: sum(call[key] for call in calls) for key in keys}
    golden = [call for call in calls if call['step'] == 'golden_experience']
    rows = [{'record_type': 'llm_call', 'case_id': case_id, **call} for call in calls]
    rows.append({'record_type': 'case_total', 'case_id': case_id, **totals,
                 'failed_calls': sum(call.get('status') in ('timeout', 'api_error') for call in calls),
                 'usage_complete': all(call.get('usage_known', True) for call in calls)})
    if golden:
        rows.append({'record_type': 'golden_experience_total', 'case_id': case_id,
                     **{key: sum(call[key] for call in golden) for key in keys}})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.writelines(json.dumps(row, ensure_ascii=False) + '\n' for row in rows)
