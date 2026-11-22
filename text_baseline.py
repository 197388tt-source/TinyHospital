"""Reproducible pure-text collaboration baseline for the competition report.

This module deliberately does *not* use TH-ACP, JSON payloads, capability
fields, or binary sidecars.  The same eight medical workflow handoffs used by
the structured benchmark are rendered as ordinary natural-language messages.
Role actions are deterministic so that communication cost, rather than model
sampling, is the variable being compared.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from statistics import mean
from time import perf_counter_ns

from agent.validation import medication_metrics
from cases import DIALMED_TRAIN_PATH, DialMedCase, load_dialmed_cases
from tool.medicine_store import MedicineKnowledgeBase
from tool.tracing import estimate_tokens


TEXT_ROUTE = (
    ("System", "Patient", "请将当前病例的真实用药私下告知专家。"),
    ("Patient", "Expert", "病例对话与真实用药如下，请保存标签以便后续评审。"),
    ("System", "Patient", "请提供不含真实标签的推荐所需信息。"),
    ("Patient", "Doctor", "请根据以下完整对话推荐恰好指定数量的药物。"),
    ("Doctor", "Expert", "以下是医生推荐，请与先前收到的真实标签比较。"),
    ("Expert", "Retrieval", "请查询推荐药物和真实药物的说明，供经验审核使用。"),
    ("Retrieval", "Expert", "以下是所请求药物的说明，请据此完成审核。"),
    ("Expert", "End", "本病例已经完成评审与经验审核。"),
)


@dataclass(frozen=True)
class TextMessage:
    sender: str
    receiver: str
    purpose: str
    body: str
    elapsed_ns: int

    @property
    def utf8_bytes(self) -> int:
        return len(self.body.encode("utf-8"))

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.body)


def _dialog(case: DialMedCase) -> str:
    try:
        turns = json.loads(case.patient_input)
    except (TypeError, ValueError):
        return case.patient_input
    return "\n".join(turns) if isinstance(turns, list) else case.patient_input


def _message(sender: str, receiver: str, purpose: str, body: str) -> TextMessage:
    started = perf_counter_ns()
    # A conventional chat-style Agent handoff repeats role, task and response
    # expectations at every boundary.  TH-ACP moves those stable semantics to
    # packet fields/capability registration; this text is intentionally kept in
    # the baseline because a receiver otherwise has no protocol vocabulary.
    rendered = (
        f"你好，{receiver}。我是 {sender}，正在协作处理一项医疗用药分析任务。\n"
        f"本轮工作：{purpose}\n"
        "请仔细阅读以下全部上下文，以自然语言完成本轮职责；不要假定你能读取其他 Agent 的内部状态。"
        "请保留病例编号和所有关键事实，完成后明确说明已处理的结果，供下一位协作者继续使用。\n\n"
        f"{body}"
    )
    return TextMessage(sender, receiver, purpose, rendered, perf_counter_ns() - started)


def _case_messages(case: DialMedCase, medicines: MedicineKnowledgeBase) -> tuple[list[TextMessage], dict]:
    """Run one text-only eight-handoff collaboration with deterministic roles."""
    dialog = _dialog(case)
    actual = list(case.drug_labels)
    recommended = list(medicines.names[:case.drug_count])
    introductions = medicines.introductions_for(list(dict.fromkeys(recommended + actual)))
    messages = [
        _message(*TEXT_ROUTE[0][:2], TEXT_ROUTE[0][2],
                 f"病例编号为 {case.case_id}。请准备处理该病例。"),
        _message(*TEXT_ROUTE[1][:2], TEXT_ROUTE[1][2],
                 f"病例编号：{case.case_id}\n完整医患对话：\n{dialog}\n\n真实药物标签：{', '.join(actual)}。"),
        _message(*TEXT_ROUTE[2][:2], TEXT_ROUTE[2][2],
                 f"病例编号为 {case.case_id}；请仅发送对话和需要推荐的药物数量。"),
        _message(*TEXT_ROUTE[3][:2], TEXT_ROUTE[3][2],
                 f"病例编号：{case.case_id}\n完整医患对话：\n{dialog}\n\n"
                 f"需要推荐 {case.drug_count} 种药物。可选药物为：{', '.join(medicines.names)}。"),
        _message(*TEXT_ROUTE[4][:2], TEXT_ROUTE[4][2],
                 f"病例编号：{case.case_id}\n医生推荐：{', '.join(recommended)}。\n"
                 "请结合你之前收到的真实药物标签进行评判。"),
        _message(*TEXT_ROUTE[5][:2], TEXT_ROUTE[5][2],
                 f"病例编号：{case.case_id}\n推荐药物：{', '.join(recommended)}。\n"
                 f"真实药物：{', '.join(actual)}。"),
        _message(*TEXT_ROUTE[6][:2], TEXT_ROUTE[6][2],
                 f"病例编号：{case.case_id}\n以下是所需药物说明：\n"
                 + "\n".join(f"{name}：{intro}" for name, intro in introductions.items())),
        _message(*TEXT_ROUTE[7][:2], TEXT_ROUTE[7][2],
                 f"病例编号：{case.case_id}\n推荐药物：{', '.join(recommended)}。\n"
                 f"真实药物：{', '.join(actual)}。\n评审已完成。"),
    ]
    metrics = medication_metrics(recommended, actual)
    return messages, {"case_id": case.case_id, "recommended_drugs": recommended,
                      "actual_drugs": actual, "exact_match": metrics["exact_match"]}


def measure(case_count: int = 10) -> dict:
    """Measure the text-only baseline on the same fixed cases as TH-ACP."""
    if case_count < 1:
        raise ValueError("case_count 必须为正数")
    cases = load_dialmed_cases(DIALMED_TRAIN_PATH, case_count)
    medicines = MedicineKnowledgeBase()
    all_messages: list[TextMessage] = []
    outcomes = []
    case_elapsed = []
    for case in cases:
        started = perf_counter_ns()
        messages, outcome = _case_messages(case, medicines)
        case_elapsed.append(perf_counter_ns() - started)
        all_messages.extend(messages)
        outcomes.append(outcome)
    by_step = []
    for step, route in enumerate(TEXT_ROUTE):
        rows = all_messages[step::len(TEXT_ROUTE)]
        by_step.append({
            "step": step, "source": route[0], "target": route[1],
            "purpose": route[2], "mean_message_bytes": round(mean(row.utf8_bytes for row in rows), 1),
            "mean_message_tokens_estimate": round(mean(row.tokens for row in rows), 1),
        })
    return {
        "mode": "pure_text", "method": "same 8-step medical workflow; natural-language messages only; deterministic role policy",
        "case_count": len(cases), "case_ids": [case.case_id for case in cases],
        "message_count": len(all_messages),
        "non_text_transfer_count": 0, "non_text_transfer_bytes": 0,
        "total_message_bytes": sum(message.utf8_bytes for message in all_messages),
        "total_message_tokens_estimate": sum(message.tokens for message in all_messages),
        "mean_case_elapsed_ms": round(mean(case_elapsed) / 1e6, 3),
        "exact_matches": sum(outcome["exact_match"] for outcome in outcomes),
        "steps": by_step,
        "outcomes": outcomes,
    }


if __name__ == "__main__":
    print(json.dumps(measure(), ensure_ascii=False, indent=2))
