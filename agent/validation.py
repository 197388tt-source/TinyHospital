"""Validation and scoring shared by Doctor and Expert."""

from __future__ import annotations

import json
import re
from collections import Counter

from tool.case_utils import _features


def _json_object(reply: str) -> dict:
    value = reply.strip()
    if value.startswith('```'):
        value = value.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    result = json.loads(value)
    if not isinstance(result, dict):
        raise ValueError('模型回复必须是 JSON 对象')
    return result


def medication_metrics(predicted: list[str], actual: list[str]) -> dict:
    if any(not isinstance(name, str) or not name.strip() for name in predicted + actual):
        raise ValueError('药物名称必须是非空字符串')
    predicted_counts, actual_counts = Counter(predicted), Counter(actual)
    matched = sum((predicted_counts & actual_counts).values())
    missing = list((actual_counts - predicted_counts).elements())
    extra = list((predicted_counts - actual_counts).elements())
    return {
        'f1': 2 * matched / (2 * matched + len(missing) + len(extra)) if predicted or actual else 0.0,
        'jaccard': matched / (matched + len(missing) + len(extra)) if predicted or actual else 0.0,
        'exact_match': not missing and not extra,
        'missing_drugs': missing,
        'extra_drugs': extra,
    }


def _validate_reasoning(reason: str, dialog: str, drugs: list[str]) -> None:
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError('判断依据不能为空')
    summary = reason.strip()
    if len(summary) > 180:
        raise ValueError('判断依据应控制在180字以内')
    if re.search(r'\[MASK\]', summary, re.I):
        raise ValueError('判断依据不能把隐藏药物占位符当作病例证据')
    if re.search(r'(?:白名单|药物名单).{0,20}(?:没有|无|缺少|不存在)'
                 r'|(?:没有|无).{0,12}药(?:物)?.{0,6}(?:可选|可用)', summary):
        raise ValueError('判断依据不能宣称白名单缺少某类药物')
    without_names = summary
    for drug in drugs:
        without_names = without_names.replace(drug, '')
    evidence = without_names
    if '关键证据' in without_names:
        evidence = without_names.split('关键证据', 1)[1].split('疾病判断', 1)[0]
    if not (_features(evidence) & _features(dialog)):
        raise ValueError('关键证据须包含当前对话中出现的症状或病程用语')
    choice = without_names.split('选药依据', 1)[1] if '选药依据' in without_names else without_names
    if len(re.findall(r'[\u4e00-\u9fffA-Za-z]', choice)) < 6:
        raise ValueError('选药依据不能只重复药物名称')


class GoldenValidationError(ValueError):
    """A stable code and actionable explanation for a rejected lesson."""
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _validate_golden_experience(answer: dict, *, dialog: str, reasoning_summary: str,
                                drugs: list[str], introductions: dict) -> str | None:
    """Validate structure, length and citation sources, not semantic support.

    reasoning_summary is retained for caller compatibility but is not inspected.
    Legacy string citations are accepted alongside arrays of source quotations.
    """
    if set(answer) != {'supported', 'evidence', 'experience'}:
        raise GoldenValidationError('response_fields', '回复只能包含 supported、evidence 和 experience')
    if type(answer.get('supported')) is not bool:
        raise GoldenValidationError('supported_type', 'supported 必须为 JSON 布尔值 true 或 false')
    evidence = answer['evidence']
    if not isinstance(evidence, list):
        raise GoldenValidationError('evidence_type', 'evidence 必须为数组')
    if not answer['supported']:
        if evidence or answer['experience'] != '':
            raise GoldenValidationError('unsupported_content', 'supported=false 时 evidence 必须为空数组，experience 必须为空字符串')
        return None
    if not evidence:
        raise GoldenValidationError('evidence_empty', 'supported=true 时 evidence 至少包含一个条目')
    for drug in drugs:
        details = introductions.get(drug)
        indication = details.get('适应症') if isinstance(details, dict) else None
        if not isinstance(indication, str) or not indication.strip():
            raise GoldenValidationError('indication_missing', f'推荐药物 {drug} 缺少有效的适应症字段')
    try:
        utterances = json.loads(dialog)
    except (TypeError, ValueError):
        utterances = dialog.splitlines()
    if not isinstance(utterances, list):
        utterances = dialog.splitlines()
    utterances = [part for part in utterances if isinstance(part, str)]
    experience = answer.get('experience')
    if not isinstance(experience, str) or not experience.strip():
        raise GoldenValidationError('experience_empty', 'experience 必须是非空字符串')
    if len(experience.strip()) > 200:
        raise GoldenValidationError('experience_too_long', f'experience 有{len(experience.strip())}个字符，超过200字符上限')
    if '\n' in experience or '\r' in experience:
        raise GoldenValidationError('experience_multiline', 'experience 不允许换行')
    lesson = experience.strip()
    seen_drugs = set()
    for entry in evidence:
        if not isinstance(entry, dict) or set(entry) != {'drug', 'case_evidence', 'drug_evidence'}:
            raise GoldenValidationError('evidence_fields', '每个 evidence 条目只能包含 drug、case_evidence 和 drug_evidence')
        drug = entry['drug']
        case_evidence = entry['case_evidence']
        drug_evidence = entry['drug_evidence']
        if isinstance(case_evidence, str):
            case_evidence = [case_evidence]
        if not isinstance(case_evidence, list) or not case_evidence:
            raise GoldenValidationError('case_evidence_empty', 'case_evidence 必须是非空原文数组')
        for quote in case_evidence:
            if not isinstance(quote, str) or not quote.strip():
                raise GoldenValidationError('case_evidence_empty', 'case_evidence 每项必须是非空字符串')
            if len(quote.strip()) < 4:
                raise GoldenValidationError('case_evidence_short', 'case_evidence 引用不足4个字符')
            if '[MASK]' in quote:
                raise GoldenValidationError('case_evidence_mask', 'case_evidence 引用不能包含 [MASK]')
            if not any(quote in part for part in utterances):
                raise GoldenValidationError('case_evidence_not_verbatim', 'case_evidence 不是医患对话中的连续原文')
        if not isinstance(drug, str) or drug not in drugs:
            raise GoldenValidationError('drug_not_recommended', 'drug 必须逐字选择医生推荐列表中的药名')
        details = introductions.get(drug)
        indication = details.get('适应症') if isinstance(details, dict) else None
        if not isinstance(indication, str) or not indication.strip():
            raise GoldenValidationError('indication_missing', '对应推荐药物缺少有效的适应症字段')
        if isinstance(drug_evidence, str):
            drug_evidence = [{'field': '适应症', 'quote': drug_evidence}]
        if not isinstance(drug_evidence, list) or not drug_evidence:
            raise GoldenValidationError('drug_evidence_empty', 'drug_evidence 必须是非空引用数组')
        cited_fields = set()
        for citation in drug_evidence:
            if not isinstance(citation, dict) or set(citation) != {'field', 'quote'}:
                raise GoldenValidationError('drug_evidence_fields', 'drug_evidence 每项只能包含 field 和 quote')
            field, quote = citation['field'], citation['quote']
            if not isinstance(field, str) or not field.strip():
                raise GoldenValidationError('drug_evidence_source', 'field 必须是对应药物说明中有效的字段名')
            source = details.get(field)
            if not isinstance(source, str) or not source.strip():
                raise GoldenValidationError('drug_evidence_source', f'对应药物说明缺少有效的 {field} 字段')
            if not isinstance(quote, str) or not quote.strip():
                raise GoldenValidationError('drug_evidence_empty', 'drug_evidence 的 quote 必须是非空字符串')
            if len(quote.strip()) < 4:
                raise GoldenValidationError('drug_evidence_short', 'drug_evidence 引用不足4个字符')
            if quote not in source:
                raise GoldenValidationError('drug_evidence_not_verbatim', f'drug_evidence 不是对应药物{field}中的连续原文')
            cited_fields.add(field)
        if '适应症' not in cited_fields:
            raise GoldenValidationError('drug_evidence_indication_missing', '每个推荐药物至少引用一项适应症')
        if drug in seen_drugs:
            raise GoldenValidationError('drug_duplicate', 'evidence 中同一药物不能重复列出')
        seen_drugs.add(drug)
    return lesson


def _validate_reflection(reflection: str, drugs: list[str], *, dialog: str,
                         reasoning_summary: str) -> None:
    """Validate only reflection type, emptiness, length and paragraph format.

    Context arguments are retained for caller compatibility but are not inspected.
    """
    if not isinstance(reflection, str):
        raise ValueError('reflection 必须是字符串；证据不足时返回空字符串')
    if reflection == '':
        return
    lesson = reflection.strip()
    if not lesson:
        raise ValueError('反思不能仅包含空白字符；证据不足时返回空字符串')
    if len(lesson) > 200:
        raise ValueError('反思应控制在200字符以内')
    if '\n' in lesson or '\r' in lesson or re.search(r'(?:^|[；;。])\s*(?:\d+[.、]|[一二三四][.、]|[-•])', lesson):
        raise ValueError('反思须写成一个段落，不能写成列表')
