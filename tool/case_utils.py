"""Dialog features and compact case summaries."""

from __future__ import annotations

import json
import re


def _features(dialog: str) -> set[str]:
    text = dialog.lower().replace('[mask]', '').replace('患者', '').replace('医生', '')
    features = set()
    for word in re.findall(r'[\u4e00-\u9fff]+|[a-z0-9]+', text):
        if re.fullmatch(r'[\u4e00-\u9fff]+', word):
            features.update(word[i:i + 2] for i in range(len(word) - 1))
            if len(word) == 1:
                features.add(word)
        else:
            features.add(word)
    return features


def _case_summary(dialog: str) -> str:
    """Keep only a short symptom, course and assessment cue, never a full dialog."""
    try:
        utterances = json.loads(dialog)
    except (TypeError, ValueError):
        utterances = [dialog]
    if not isinstance(utterances, list):
        utterances = [dialog]
    utterances = [part.strip() for part in utterances if isinstance(part, str) and part.strip()]
    patients = [part.removeprefix('患者：').strip() for part in utterances
                if part.startswith('患者：')]
    doctors = [part.removeprefix('医生：').strip() for part in utterances
               if part.startswith('医生：')]
    symptom = (patients or utterances or ['病例信息不足'])[0][:24]
    duration = re.compile(r'好几天|前几天|半个月|\d+\s*(?:小时|个月|天|周|月|年)'
                          r'|[一二三四五六七八九十两半几]+(?:小时|个月|天|周|月|年)'
                          r'|昨天|前天|最近')
    course = next((match.group() for part in patients
                   if (match := duration.search(part)) and match.group() not in symptom), '')
    assessment = next((part[:18] for part in doctors
                       if re.search(r'考虑|可能|确诊|诊断|属于|应该是', part)), '')
    if not assessment:
        assessment = '病因待确认'
    pieces = [part for part in (symptom, course, assessment) if part]
    return '；'.join(pieces)[:50]
