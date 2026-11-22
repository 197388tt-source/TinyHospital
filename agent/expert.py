"""Expert agent: compare the submitted drugs with private labels and reflect."""

from __future__ import annotations

import json
from time import perf_counter

from tool.diagnostics import emit
from typing import Callable

from tool.llm_client import LLMClient
from tool.medicine_store import MedicineKnowledgeBase
from history.memory import CaseMemoryStore
from tool.tracing import Trace

from .base import ModelAgent, State
from .validation import (_json_object, _validate_golden_experience,
                         _validate_reflection, medication_metrics)


class ExpertAgent(ModelAgent):
    def __init__(self, trace: Trace, memories: CaseMemoryStore, medicines: MedicineKnowledgeBase,
                 dataset_path: str, llm_client_factory: Callable[[], LLMClient] = LLMClient) -> None:
        super().__init__('ExpertAgent', trace, llm_client_factory)
        self.memories, self.medicines, self.dataset_path = memories, medicines, dataset_path
        self.clear_case()

    def clear_case(self) -> None:
        self.case: dict | None = None
        self.evaluation: dict | None = None
        self.clear_model_calls()
        self.review_events = []

    def _review_event(self, stage, status, reason=None, **extra):
        event = {'event': 'expert_review', 'case_id': self.case['case_id'],
                 'trace_id': getattr(self.runtime, 'trace_id', None),
                 'stage': stage, 'status': status, **extra}
        if reason is not None:
            event['reason'] = reason
        self.review_events.append(event)
        emit(self.memories.diagnostics_path, event)
        if self.memories.diagnostics_verbose:
            print(f"[专家评审] 病例 {self.case['case_id']} {stage}: {status}"
                  + (f"；{reason}" if reason else ''), flush=True)

    def receive_case_truth(self, message: str) -> str:
        self.receive_task(State.IDLE)
        data = json.loads(message)
        if self.case is not None or not data.get('patient_dialog') or not data.get('actual_drugs'):
            raise ValueError('专家病例或真实药物无效')
        self.case = data
        self.transition(State.IDLE)
        return '专家已接收私有病例标签。'

    def evaluate(self, message: str) -> str:
        self.receive_task(State.IDLE)
        data = json.loads(message)
        if self.case is None or self.evaluation is not None or data.get('case_id') != self.case['case_id']:
            raise RuntimeError('医生推荐与当前病例不匹配')
        drugs = data.get('recommended_drugs')
        if (not isinstance(drugs, list) or len(drugs) != len(self.case['actual_drugs'])
                or len(drugs) != len(set(drugs)) or any(name not in self.medicines.names for name in drugs)):
            raise ValueError('医生推荐不符合数量或白名单约束')
        if self.memories.training and (
                not isinstance(data.get('reasoning_summary'), str) or not data['reasoning_summary'].strip()):
            raise ValueError('医生推荐缺少判断依据')
        metrics = medication_metrics(drugs, self.case['actual_drugs'])
        self.evaluation = {**data, 'metrics': metrics}
        self.transition(State.IDLE)
        outcome = {'case_id': data['case_id'], 'recommended_drugs': drugs,
                   'actual_drugs': self.case['actual_drugs']}
        if not self.memories.training:
            outcome.update(exact_match=metrics['exact_match'],
                           reflection=None, golden_experience=None)
        return json.dumps(outcome, ensure_ascii=False)

    def review(self, message: str) -> str:
        self.receive_task(State.IDLE)
        details = json.loads(message)
        recommended = details.get('recommended_drugs')
        actual = details.get('actual_drugs')
        introductions = details.get('medicine_introductions')
        if (self.case is None or self.evaluation is None
                or details.get('case_id') != self.case['case_id']
                or recommended != self.evaluation['recommended_drugs']
                or actual != self.case['actual_drugs']
                or not isinstance(introductions, dict)
                or set(introductions) != set(recommended) | set(actual)):
            raise RuntimeError('药物说明与当前专家病例不匹配')
        details['recommended_introductions'] = {
            name: introductions[name] for name in recommended
        }
        details['actual_introductions'] = {
            name: introductions[name] for name in actual
        }
        case, evaluation = self.case, self.evaluation
        metrics = evaluation['metrics']
        reflection = None
        reflection_error = None
        golden_experience = None
        if metrics['exact_match']:
            details['recommended_introductions'] = {
                name: {field: value for field, value in introduction.items()
                       if field != '不良反应'}
                for name, introduction in details['recommended_introductions'].items()
            }
            instruction = (
                '你是一名医学专家，请仅依据医患对话、医生推荐药物、判断依据及药物说明，审核并总结一条可复用的正向选药经验。\n'
                '\n'
                '任务一、审核\n'
                '医生推荐药物与参考答案一致，不代表医生判断依据正确或治疗有效。\n'
                '核查医生判断依据中的病例事实、疾病判断和全部推荐药物的选药理由：\n'
                '- 病例事实须有对话支持，疾病判断须保留原有不确定性。\n'
                '- 每个推荐药物均须有明确理由，符合至少一项适应症及限定条件，且不与材料中明确的用药限制冲突。\n'
                '- 不得补充事实、将未提及视为不存在，或替医生补写缺失的理由。\n'
                '\n'
                '全部通过后，初步判定 supported=true，依次完成任务二、任务三，并在最终检查通过后输出；任一项不满足或必要证据不足，supported=false，跳过任务二、任务三，直接进行任务四。\n'
                '\n'
                '任务二、总结\n'
                '围绕一个核心选药问题，按以下顺序写成一个连贯段落：\n'
                '1. 病人状况：关键症状、相关病史，以及影响选药的既往用药和治疗反应。\n'
                '2. 可能的问题：材料支持的疑似疾病或功能问题，证据不足时省略。\n'
                '3. 选药及理由：可考虑哪些推荐药物、改善什么问题、为什么；有依据时区分继续使用、加用或替换。\n'
                '4. 注意事项：与该选择直接相关、材料支持的限制或待核实信息，无依据时不强行补写。\n'
                '\n'
                '不必覆盖全部推荐药物，但须保留经验成立的关键条件。\n'
                '不得将适应症匹配推导为疗效保证、首选优势或联用获益，\n'
                '也不得将信息缺失写成无禁忌或无相互作用。\n'
                '经验须能独立理解，建议100到150字符，最多200个字符（含标点）。不得为缩短文字省略必要限定。\n'
                '\n'
                '任务三、引用证据\n'
                '为经验提出选药建议的每个药物提供一个 evidence 条目，同一药物不重复列出：\n'
                '- drug：与医生推荐列表中的药名一致。\n'
                '- case_evidence：对话原文数组，每项连续摘自单条发言，不含 [MASK]。\n'
                '- drug_evidence：药物说明引用数组，每项包含 field 和 quote；至少引用一项适应症，涉及用药限制时补充对应字段。\n'
                '\n'
                '每段引用至少4个字符，不得改写、拼接或遗漏关键否定词和限定条件。\n'
                '引用须共同支持经验中的病例条件、判断、选药理由和注意事项。\n'
                '既往药物仅作背景时无需单独建立条目，但其使用情况须有对话证据。\n'
                '\n'
                '无法形成有依据的经验，或无法提供必要证据时，supported=false。\n'
                '\n'
                '任务四、输出\n'
                '只返回合法 JSON，仅包含 supported、evidence、experience，不输出分析过程或 Markdown。\n'
                'supported 为布尔值，evidence 为数组，experience 为字符串。\n'
                'supported=true 时，evidence 至少包含一个有效条目，experience 非空。\n'
                '\n'
                '通过：\n'
                '{"supported":true,"evidence":[{"drug":"药名","case_evidence":["对话原文"],"drug_evidence":[{"field":"适应症","quote":"说明原文"}]}],"experience":"正向选药经验"}\n'
                '\n'
                '未通过：\n'
                '{"supported":false,"evidence":[],"experience":""}'
            )
            context = (
                '医患对话：' + case['patient_dialog']
                + '\n医生判断依据：' + evaluation['reasoning_summary']
                + '\n医生推荐药物：' + json.dumps(evaluation['recommended_drugs'], ensure_ascii=False)
                + '\n推荐药物说明：' + json.dumps(details['recommended_introductions'], ensure_ascii=False)
            )
            base_context = context
            for attempt in range(3):
                reply = None
                answer = None
                try:
                    reply = self.ask_llm('golden_experience', instruction, context)
                    answer = _json_object(reply)
                    golden_experience = _validate_golden_experience(
                        answer, dialog=case['patient_dialog'],
                        reasoning_summary=evaluation['reasoning_summary'],
                        drugs=evaluation['recommended_drugs'],
                        introductions=details['recommended_introductions'],
                    )
                    self._review_event(
                        'golden_experience',
                        'accepted' if golden_experience is not None else 'unsupported',
                        None if golden_experience is not None else '模型明确表示证据不足',
                        attempt=attempt + 1, raw_reply=reply, parsed_reply=answer,
                    )
                    break
                except (RuntimeError, TimeoutError, OSError) as exc:
                    status = 'timeout' if isinstance(exc, TimeoutError) else 'api_error'
                    self._review_event('golden_experience', status, type(exc).__name__,
                                       attempt=attempt + 1, raw_reply=reply, parsed_reply=answer,
                                       error_code='api_error')
                    break
                except (KeyError, TypeError, ValueError) as exc:
                    # Transport/API-format errors are not content-correction attempts.
                    failed_call = self.model_calls and self.model_calls[-1].get('status') != 'returned'
                    self._review_event(
                        'golden_experience', 'api_error' if failed_call else 'validation_rejected',
                        type(exc).__name__ if failed_call else str(exc), attempt=attempt + 1,
                        raw_reply=reply, parsed_reply=answer,
                        error_code=('api_error' if failed_call else
                                    getattr(exc, 'code', 'reply_format_invalid')),
                    )
                    if failed_call or attempt == 2:
                        break
                    context = (
                        base_context
                        + '\n\n上一版回复：\n' + reply[:2000]
                        + '\n校验未通过：' + str(exc)
                        + '\n请依据原始材料修正上述问题，并重新返回完整 JSON 对象。'
                        + '\ncase_evidence 数组中的每项及 drug_evidence 数组中的 quote 必须连续复制对应原文，'
                          '不得改写、拼接，也不得选取不相关内容来通过校验。'
                        + '\n如果材料确实不能支持经验，请返回 supported=false，'
                          'evidence 置为空数组，experience 置为空字符串。'
                    )

        else:
            instruction = (
                '你是一名医学专家，负责判断药物推荐是否存在需要纠正的问题。只有存在充分依据的问题时，才提炼可供相似病例参考的反思经验。\n'
                '\n'
                '你将获得医患对话、医生推荐药物、医生判断依据、参考推荐药物、相关药物说明及标签差异。\n'
                '\n'
                '参考推荐来自数据集标签，不代表唯一合理方案。标签差异不能直接视为临床错误，参考药物也不能直接视为临床正确答案。\n'
                '\n'
                '一、逐药审核与整体评估\n'
                '\n'
                '先根据对话识别患者的主要症状、关键问题、现用药及明确风险，再审核完整推荐方案，最后比较参考药物。\n'
                '\n'
                '1. 对医生推荐的每一种药物，判断其是否有与当前症状、疾病判断或用药场景相对应的依据，以及是否与明确的用药限制冲突。与参考答案相同的药物也必须审核，不能自动视为合理。\n'
                '\n'
                '2. 对“推荐有、参考无”的药物，逐一判断：\n'
                '   - 有合理使用依据，且没有明确冲突：仅属于标签差异，不作为错误。\n'
                '   - 有充分依据认定推荐不合理：记录具体药名、对话依据及不合理原因，作为可反思的问题。\n'
                '   - 必要信息不足，无法判断：保留不确定性，不直接认定为错误。\n'
                '\n'
                '3. 对“参考有、推荐无”的药物，逐一判断：\n'
                '   - 该药针对什么治疗目标，对话是否支持这一目标。\n'
                '   - 医生现有方案是否已经合理覆盖该目标。\n'
                '   - 未推荐该药是否意味着遗漏了会明显改变选药或处理方式的关键问题。\n'
                '   如果现有方案已合理覆盖该目标，或该药只是可选治疗，不认定为遗漏。\n'
                '   只有确认存在关键遗漏时，才将该问题作为反思依据；只有材料支持时，才提出该药作为可考虑的处理，不能仅凭参考答案要求补药。\n'
                '   如果关键问题需要先补充评估，可以反思评估不足，不必强行增加药物。\n'
                '\n'
                '4. 逐药审核后，结合完整方案检查治疗目标是否覆盖、现用药是否存在冲突，以及推荐是否只处理了次要症状。不得将逐药审核变成逐药补齐参考答案。\n'
                '\n'
                '二、判断是否需要生成反思\n'
                '\n'
                '只有满足以下至少一项，才可以生成反思：\n'
                '\n'
                '1. 推荐药物缺乏与当前症状、疾病判断或用药场景相对应的依据，或与材料明确的症状、禁忌、现用药及用药风险明显冲突。\n'
                '2. 推荐遗漏了对话中已明确、且会明显改变选药或处理方式的关键症状或问题，导致只处理次要症状。遗漏的处理可以是用药，也可以是必要的补充评估。\n'
                '\n'
                '以下情况不生成反思：\n'
                '- 推荐与参考药物不同，但同样具有合理依据，且没有上述问题。\n'
                '- 只能确认标签差异，不能确认具体的用药问题或关键遗漏。\n'
                '- 仅因为参考列表包含某药，就认为必须补充该药。\n'
                '- 仅因为还可以增加问诊、检查或注意事项，就认为原推荐有错。\n'
                '- 必要材料缺失，无法作出有依据的判断。\n'
                '\n'
                '“还可以进一步核查”本身不是生成反思的理由。核查建议只有在对应一个已由对话触发、会改变处理决策的关键问题时，才能进入反思。\n'
                '\n'
                '三、判断时遵守的边界\n'
                '\n'
                '1. 不得补充病例事实，不得将医生的问句视为患者事实，不得将“未提及”视为“不存在”，不得将疑似诊断写成确诊。\n'
                '2. 区分“对话没有提供某药的使用依据”和“药物说明资料缺失”。后者不等于该药不适用，也不能仅因信息不足就认定推荐错误。\n'
                '3. 区分禁用、慎用、不推荐合用和需要监测。不良反应中列出某症状，不等于该药对有此症状的患者绝对禁用；应结合使用依据和风险判断。\n'
                '4. 不得把所有症状都映射成必须增加一种药。已有方案可能覆盖多个症状，部分问题需要评估而非加药。\n'
                '5. 区分继续使用、加用、换药和既往已停药。再次提及同一药物不自动等于重复加量，作用相近也不自动等于禁止联用。不得默认模型方案、参考方案与既往药物同时使用。\n'
                '6. 不得仅因简短的医生判断依据没有展开某项常规评估，就断言医生遗漏了该问题。\n'
                '7. 无法确认存在具体用药问题或关键遗漏时，返回空反思，不为了生成经验寻找纠错理由。\n'
                '\n'
                '四、提炼可复用反思\n'
                '\n'
                '完成全部药物审核后，围绕一个有充分依据、影响选药或处理决策的核心问题提炼经验。若存在多个独立问题，优先选择风险更高或更影响主要问题处理的一项；不必把完整审核过程写入反思。\n'
                '\n'
                '采用以下结构：\n'
                '“如果患者具备【触发问题的关键条件】，不应【具体不合理推荐或遗漏的处理方式】，因为【具体原因】；应【有依据的纠正原则或必要评估】。”\n'
                '\n'
                '写作要求：\n'
                '\n'
                '1. 以“如果……”或“当……时……”开头，写清关键条件如何影响选药，不复述完整病例，不使用“本例”“上述患者”等指代。\n'
                '2. 所选核心问题涉及已确认错误推荐的药物时，必须逐一写出与推荐列表一致的药名及不合理原因，不用“某药”“这类药”或“原方案”代替。\n'
                '3. 合理的标签差异不写入纠错内容。只有确认关键遗漏且材料支持时，才提及未推荐的参考药物及其适用条件；替代、加用或停用建议也须有依据。\n'
                '4. 补充评估须说明其对应的关键问题及如何影响处理，不能泛泛写“先核实再决定”。\n'
                '5. 保留必要限定与不确定性，不把个案判断扩大成某药普遍适用或不适用。\n'
                '6. 一个段落，通常一到两句，最多200字符，不设最低字数；不为压缩文字省略错误药名、关键否定词或必要限定。\n'
                '7. 若只能形成病例摘要、一般性提醒或没有明确决策意义的核查建议，返回空反思。\n'
                '\n'
                '五、输出\n'
                '\n'
                '仅返回合法 JSON 对象，仅包含 reflection 字段；reflection 为字符串，不输出审核过程或 Markdown。\n'
                '\n'
                '存在有依据的问题且能提炼经验时：\n'
                '{"reflection":"如果……，不应……；应……。"}\n'
                '\n'
                '推荐合理、仅有标签差异、证据不足或无法提炼经验时：\n'
                '{"reflection":""}'
            )
            context = (
                '医患对话：' + case['patient_dialog']
                + '\n医生推荐依据：' + evaluation['reasoning_summary']
                + '\n医生推荐药物：' + json.dumps(evaluation['recommended_drugs'], ensure_ascii=False)
                + '\n医生推荐药物说明：' + json.dumps(details['recommended_introductions'], ensure_ascii=False)
                + '\n参考推荐药物说明：' + json.dumps(details['actual_introductions'], ensure_ascii=False)
                + '\n参考推荐药物：' + json.dumps(case['actual_drugs'], ensure_ascii=False)
                + '\n标签差异（参考推荐有、医生推荐无）：' + json.dumps(metrics['missing_drugs'], ensure_ascii=False)
                + '\n标签差异（医生推荐有、参考推荐无）：' + json.dumps(metrics['extra_drugs'], ensure_ascii=False)
            )
            base_context = context
            for attempt in range(3):
                value = None
                try:
                    answer = _json_object(self.ask_llm('reflect', instruction, context))
                    if set(answer) != {'reflection'}:
                        raise ValueError('专家回复只能包含 reflection')
                    value = answer['reflection']
                    _validate_reflection(
                        value, evaluation['recommended_drugs'] + case['actual_drugs'],
                        dialog=case['patient_dialog'],
                        reasoning_summary=evaluation['reasoning_summary'],
                    )
                    reflection = value.strip() or None
                    self._review_event(
                        'reflection', 'accepted' if reflection is not None else 'unsupported',
                        None if reflection is not None else '模型明确表示证据不足',
                        attempt=attempt + 1,
                    )
                    break
                except (RuntimeError, TimeoutError, OSError) as exc:
                    status = 'timeout' if isinstance(exc, TimeoutError) else 'api_error'
                    self._review_event('reflection', status, type(exc).__name__, attempt=attempt + 1)
                    raise  # Preserve explicit transport failure behavior.
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    self._review_event('reflection', 'validation_rejected', str(exc), attempt=attempt + 1)
                    if attempt == 2:
                        reflection_error = f'专家连续 3 次返回无效反思：{exc}'
                        break
                    feedback = f'\n上一回复未通过校验：{exc}。'
                    if isinstance(value, str) and len(value.strip()) > 200:
                        feedback += (f'上一条反思有{len(value.strip())}个字符，超过200字符上限。'
                                     '请压缩至200个字符以内，不设最低字数；围绕一个核心选药问题，通常一到两句。以“如果……”或“当……时……”明确适用条件。'
                                     '保留关键病例条件、所选核心问题涉及的错误药名、不合理原因及必要限制，'
                                     '删除重复背景和分析过程。'
                                     '\n待改写反思（最多展示1000字符）：' + value[:1000])
                    feedback += ('\n所选核心问题涉及错误推荐时，逐一写出对应准确药名及原因；只有确认关键遗漏且材料支持时才提及参考药物，不要求两侧同时出现。'
                                 '\n只返回 {"reflection":"可复用的纠错经验"}，'
                                 '证据不足时返回 {"reflection":""}；推荐合理、仅有标签差异或无法提炼纠错经验时也返回空反思，不要输出其他内容。')
                    context = base_context + feedback
        outcome = {
            'case_id': case['case_id'],
            'recommended_drugs': evaluation['recommended_drugs'],
            'actual_drugs': case['actual_drugs'],
            'exact_match': metrics['exact_match'],
            'reflection': reflection,
            'golden_experience': golden_experience,
        }
        if reflection_error is not None:
            outcome['reflection_error'] = reflection_error
        saving_started = perf_counter()
        try:
            if metrics['exact_match']:
                self.memories.save_success(case['case_id'], golden_experience)
            elif reflection is not None:
                self.memories.save_reflection(case['case_id'], reflection)
        except Exception as exc:
            self._review_event('memory_sync', 'failed', type(exc).__name__,
                               elapsed_ms=round((perf_counter() - saving_started) * 1000, 3))
            raise
        self._review_event('memory_sync',
                           'frozen' if not self.memories.training else
                           'saved' if golden_experience is not None or reflection is not None else 'no_new_lesson',
                           elapsed_ms=round((perf_counter() - saving_started) * 1000, 3))
        self.transition(State.IDLE)
        return json.dumps(outcome, ensure_ascii=False)
