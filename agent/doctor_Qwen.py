"""Doctor agent: train with reasoning; test with historical lessons."""

from __future__ import annotations

import json
from tool.diagnostics import emit
from typing import Callable

from tool.llm_client import LLMClient
from tool.medicine_store import MedicineKnowledgeBase
from tool.tracing import Trace

from .base import State
from .doctor import DoctorAgent as APIDoctorAgent
from .validation import _json_object, _validate_reasoning


class DoctorAgent(APIDoctorAgent):



    def _baseline_instruction(self, count: int) -> str:
        return self._training_instruction(count)

    def _test_instruction(self, count: int, experience_count: int) -> str:
        return (
            self._baseline_instruction(count) + '\n'
            f"下面是当前医患对话，以及从相似历史医患对话中提取的{experience_count}条药物推荐经验。"
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
            experience_count = len(history['golden_experiences']) + len(history['reflections'])
            instruction = self._test_instruction(count, experience_count)
            context = dialog + '\n相关历史经验：' + json.dumps(history, ensure_ascii=False)
        else:
            instruction = self._baseline_instruction(count)
            context = dialog
        invalid_attempts = []
        for attempt in range(3):
            reply = None
            try:
                reply = self.ask_llm('recommend', instruction, context)
                answer = _json_object(reply)
                if set(answer) != {'recommended_drugs', 'reasoning_summary'}:
                    raise ValueError('医生回复必须且只能包含 recommended_drugs 和 reasoning_summary')
                drugs, reason = answer['recommended_drugs'], answer['reasoning_summary']
                if (not isinstance(drugs, list) or len(drugs) != count
                        or any(not isinstance(name, str) or name not in self.medicines.names for name in drugs)
                        or len(set(drugs)) != len(drugs)):
                    raise ValueError(f'必须推荐恰好 {count} 种互不重复的白名单药物')
                _validate_reasoning(reason, dialog, drugs)
                self.recommended_drugs = tuple(drugs)
                self.transition(State.IDLE)
                recommendation = {'case_id': data['case_id'], 'recommended_drugs': drugs}
                if self.training:
                    recommendation['reasoning_summary'] = reason.strip()
                return json.dumps(recommendation, ensure_ascii=False)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                invalid_attempts.append({'attempt': attempt + 1, 'raw_reply': reply,
                                         'error': str(exc)})
                emit(getattr(self, 'diagnostics_path', None), {
                    'event': 'doctor_validation_rejected', 'case_id': self.case_id,
                    'trace_id': getattr(self.runtime, 'trace_id', None),
                    'attempt': attempt + 1, 'reason': str(exc), 'raw_reply': reply})
                if getattr(self, 'diagnostics_verbose', False):
                    print(f'[医生重试] 病例 {self.case_id} 第{attempt + 1}次输出无效：{exc}', flush=True)
                if attempt == 2:
                    self.transition(State.IDLE)
                    return json.dumps({
                        'case_id': self.case_id, 'status': 'doctor_failed',
                        'recommended_drugs': None,
                        'error': f'医生连续 3 次返回无效内容：{exc}',
                        'attempts': invalid_attempts,
                    }, ensure_ascii=False)
                context += f'\n上一回复无效：{exc}。请严格按指定 JSON 格式重答。'
        raise AssertionError('不可达')
