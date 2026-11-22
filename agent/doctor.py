"""Doctor agent: train with reasoning; test with historical lessons."""

from __future__ import annotations

import json
from tool.diagnostics import emit
from typing import Callable

from tool.llm_client import LLMClient
from tool.medicine_store import MedicineKnowledgeBase
from tool.tracing import Trace

from .base import ModelAgent, State
from .validation import _json_object, _validate_reasoning


class DoctorAgent(ModelAgent):
    def __init__(self, trace: Trace, medicines: MedicineKnowledgeBase,
                 llm_client_factory: Callable[[], LLMClient] = LLMClient, *,
                 training: bool = True) -> None:
        if type(training) is not bool:
            raise ValueError('training 必须为布尔值')
        super().__init__('DoctorAgent', trace, llm_client_factory)
        self.training = training
        self.medicines = medicines
        self.clear_case()

    def clear_case(self) -> None:
        self.case_id = None
        self.recommended_drugs: tuple[str, ...] = ()
        self.clear_model_calls()

    def _training_instruction(self, count: int) -> str:
        return (
            "你是一名专业的医生，正在完成医患对话药物推荐任务。"
            "对话中的 [MASK] 表示被隐藏的用药建议。"
            "请根据完整医患对话，"
            f"从药物白名单中推荐恰好 {count} 种互不重复的药物。\n"

            "推荐要求：\n"
            "1. 综合患者明确陈述的症状、病程、既往用药及治疗反馈，"
            "参考对话中医生已有的疾病判断；医生的问句不能当作已确认的患者事实。\n"

            "2. 重点考虑药物与疾病判断、主要症状及用药场景的匹配程度。"
            "若对话中的医生已有倾向性病因，应优先检查白名单中与该病因一致的药物，"
            "不要只凭瘙痒等共同症状选择对症药。"
            "先核对疾病判断与所选药物作用是否一致；仅缓解症状的药物不能写成针对病因的治疗，"
            "也不能用对症效果反过来证明某个病因。"
            "对于对话中明确提及的特殊人群、禁忌及用药风险，应予以考虑。\n"

            "3. 不能编造对话中没有的检查、病史、过敏或特殊人群信息，"
            "也不得将尚未明确的疾病判断当作确诊。"
            "不要在缺少依据时断言某药对当前患者绝对安全。\n"

            "请同时提供 reasoning_summary，控制在180字以内，严格按一行三段书写："
            "关键证据：对话中的症状或病程；疾病判断：考虑或明确的疾病及必要的不确定性；"
            "选药依据：所选药物与当前判断或症状的对应理由。"
            "不要简单重复药名、复述完整对话、把 [MASK] 当作证据，"
            "也不要宣称白名单缺少某类药物。\n"

            "仅输出合法 JSON 对象，且只能包含 recommended_drugs 与 reasoning_summary 两个字段。"
            "recommended_drugs 应为恰好符合目标数量的药名数组。"
            "不要输出其他文字或 Markdown。\n"

            "药物名称必须与白名单完全一致。\n"
            "药物白名单：" + "、".join(self.medicines.names)
        )

    def _baseline_instruction(self, count: int) -> str:
        return (
            "你是一名医生，你的任务是根据下面的医患对话进行药物推荐。"
            f"恰好输出 {count} 个药名，必须逐字从以下白名单中选择：{'、'.join(self.medicines.names)}。"
            '只输出 JSON 字符串数组，例如 ["药名一","药名二"]；不要解释、编号或 Markdown。'
        )

    def _test_instruction(self, count: int) -> str:
        return (
            self._baseline_instruction(count) + '\n'
            "你可能收到以下两类从历史病例中提炼的参考经验：\n"
            "- 正向经验（golden_experiences）：总结历史病例中有依据的选药原则，"
            "用于参考哪些病例条件支持某种选药，以及相应理由。\n"
            "- 错误反思（reflections）：总结历史推荐中值得纠正或核查的问题，"
            "用于提示需要检查的选药依据、适用条件或用药限制。"
            "反思中的药物取舍只适用于其所述条件，不代表该药物普遍适用或不适用。\n"
            "任一类经验都可能为空，两类也可能同时为空。仅参考实际提供的内容；"
            "为空表示本次未提供该类经验，不代表当前病例没有合理选药方案或不存在需要核查的问题。"
            "两类均为空时，依据当前医患对话和药物白名单完成推荐。\n"
            "使用经验前，先核对其关键条件是否得到当前医患对话支持。"
            "条件符合时，可以参考其中的选药或核查原则；条件不明时，不能假定符合；"
            "经验不相关或与当前病例事实冲突时，不采用该经验。"
            "正向经验和错误反思均不是当前医患对话原文，也不是当前病例的标准答案。"
            "不得将其中的症状、诊断、病史或治疗反馈当作当前患者事实，"
            "不得表述为当前患者或医生说过的话。"
        )

    def recommend(self, message: str) -> str:
        self.receive_task(State.IDLE)
        data = json.loads(message)
        self.case_id = data.get('case_id')
        count, dialog = data['count'], data['patient_dialog']
        history = data.get('history', {'golden_experiences': [], 'reflections': []})
        if (type(data.get('case_id')) is not int
                or 'actual_drugs' in data or 'drug_labels' in data
                or type(count) is not int or not 0 < count <= len(self.medicines.names)
                or not isinstance(dialog, str) or not dialog.strip()
                or (not self.training and (
                    not isinstance(history, dict) or set(history) != {'golden_experiences', 'reflections'}
                    or any(not isinstance(items, list)
                           or any(not isinstance(item, str) or not item.strip() for item in items)
                           for items in history.values())))):
            raise ValueError('医生收到的病例信息无效')
        if self.training:
            instruction = self._training_instruction(count)
            context = '完整医患对话：' + dialog + f'\n目标药物数量：{count}'
        elif history['golden_experiences'] or history['reflections']:
            instruction = self._test_instruction(count)
            context = dialog + '\n相关历史经验：' + json.dumps(history, ensure_ascii=False)
        else:
            instruction = self._baseline_instruction(count)
            context = dialog
        for attempt in range(3):
            try:
                reply = self.ask_llm('recommend', instruction, context)
                if self.training:
                    answer = _json_object(reply)
                    if set(answer) != {'recommended_drugs', 'reasoning_summary'}:
                        raise ValueError('医生回复必须且只能包含 recommended_drugs 和 reasoning_summary')
                    drugs, reason = answer['recommended_drugs'], answer['reasoning_summary']
                else:
                    drugs = json.loads(reply)
                if (not isinstance(drugs, list) or len(drugs) != count
                        or any(not isinstance(name, str) or name not in self.medicines.names for name in drugs)
                        or len(set(drugs)) != len(drugs)):
                    raise ValueError(f'必须推荐恰好 {count} 种互不重复的白名单药物')
                if self.training:
                    _validate_reasoning(reason, dialog, drugs)
                self.recommended_drugs = tuple(drugs)
                self.transition(State.IDLE)
                recommendation = {'case_id': data['case_id'], 'recommended_drugs': drugs}
                if self.training:
                    recommendation['reasoning_summary'] = reason.strip()
                return json.dumps(recommendation, ensure_ascii=False)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                emit(getattr(self, 'diagnostics_path', None), {
                    'event': 'doctor_validation_rejected', 'case_id': self.case_id,
                    'trace_id': getattr(self.runtime, 'trace_id', None),
                    'attempt': attempt + 1, 'reason': str(exc)})
                if getattr(self, 'diagnostics_verbose', False):
                    print(f'[医生重试] 病例 {self.case_id} 第{attempt + 1}次输出无效：{exc}', flush=True)
                if attempt == 2:
                    raise RuntimeError(f'医生连续 3 次返回无效内容：{exc}') from exc
                context += f'\n上一回复无效：{exc}。请严格按指定 JSON 格式重答。'
        raise AssertionError('不可达')
