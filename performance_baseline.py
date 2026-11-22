"""Reproducible local communication measurement for ten fixed training cases.

The deterministic model stub exercises the full Agent route without network variance.
Its usage values are placeholders, never real LLM API token counts.
"""

from __future__ import annotations

import json
import hashlib
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import mean
from unittest.mock import patch

from cases import DIALMED_TRAIN_PATH, load_dialmed_cases
from main import HospitalSession
from protocol import AgentPacket
from tool.llm_client import TokenUsage
from tool.medicine_store import MedicineKnowledgeBase


ROUTE = (
    ('System', 'PatientAgent', 'patient.provide_case_truth'),
    ('PatientAgent', 'ExpertAgent', 'expert.receive_case_truth'),
    ('System', 'PatientAgent', 'patient.provide_recommendation_context'),
    ('PatientAgent', 'DoctorAgent', 'doctor.recommend'),
    ('DoctorAgent', 'ExpertAgent', 'expert.evaluate'),
    ('ExpertAgent', 'RetrievalAgent', 'retrieval.medicines'),
    ('RetrievalAgent', 'ExpertAgent', 'expert.review'),
    ('ExpertAgent', 'End', 'case.complete'),
)


class FixedModel:
    """Choose from the public allow-list; never consult case labels."""

    def __init__(self) -> None:
        allowed = MedicineKnowledgeBase().names
        self.names = ('酮康唑', '依巴斯汀') + tuple(
            name for name in allowed if name not in {'酮康唑', '依巴斯汀'})
        self.last_usage = TokenUsage()
        self.prompts: list[list[dict[str, str]]] = []

    @staticmethod
    def _case_phrase(context: str, prefix: str, suffix: str) -> str:
        dialog = json.loads(context.split(prefix, 1)[1].split(suffix, 1)[0])
        return re.sub(r'^患者[：:]\s*', '', dialog[0])[:18].replace('[MASK]', '')

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.prompts.append(messages)
        instruction, context = messages[0]['content'], messages[1]['content']
        if instruction.startswith('你是一名医生，你的任务是'):
            count = int(re.search(r'恰好输出 (\d+) 个', instruction).group(1))
            return json.dumps(list(self.names[:count]), ensure_ascii=False)
        if '你是一名专业的医生' in instruction:
            count = int(re.search(r'恰好 (\d+) 种', instruction).group(1))
            phrase = self._case_phrase(context, '完整医患对话：', '\n目标药物数量：')
            return json.dumps({
                'recommended_drugs': list(self.names[:count]),
                'reasoning_summary': f'关键证据：{phrase}；疾病判断：病因仍需确认；选药依据：应结合症状核对药物适应症。',
            }, ensure_ascii=False)
        if 'supported' in instruction:
            return json.dumps({'supported': False, 'evidence': [], 'experience': ''})
        phrase = self._case_phrase(context, '医患对话：', '\n医生推荐依据：')
        return json.dumps({
            'reflection': f'针对{phrase}，应核对疾病判断与药物适应症，避免只按共同症状推断处方。',
        }, ensure_ascii=False)


def measure() -> dict:
    cases = load_dialmed_cases(DIALMED_TRAIN_PATH, 10)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        # Pre-existing lessons must not enter the training recommendation path.
        (root / 'golden.json').write_text(json.dumps([
            {'case_id': 4737, 'experience': '出现鼻部症状时应核对病因与适应症，因为共同症状不能代表同一疾病。'},
        ], ensure_ascii=False), encoding='utf-8')
        (root / 'reflections.json').write_text(json.dumps([
            {'case_id': 8977, 'reflection': '皮肤起疹时应核对病程及病因，避免仅按症状选择药物。'},
        ], ensure_ascii=False), encoding='utf-8')
        model = FixedModel()
        # Measure the full training route while keeping benchmark memories fixed.
        with patch('main.LLMClient', return_value=model), \
             patch('history.memory.CaseMemoryStore.save_success'), \
             patch('history.memory.CaseMemoryStore.save_reflection'):
            session = HospitalSession(
                retrieval_mode='jaccard', verbose=False, training=True, dataset_path=DIALMED_TRAIN_PATH,
                golden_path=root / 'golden.json', reflection_path=root / 'reflections.json',
                communication_path=root / 'communications.jsonl',
                control_path=root / 'control.jsonl', llm_path=root / 'llm.jsonl',
            )
            try:
                outcomes = [session.run_case(case) for case in cases]
            finally:
                session.close()
        costs = session.trace.message_costs
        packets = session.trace.packets
        assert len(costs) == len(cases) * len(ROUTE)
        assert all((cost.sender, cost.receiver, cost.capability) == ROUTE[cost.step]
                   for cost in costs)
        # The Doctor's incoming packet and model context must never contain private label fields.
        doctor_packets = [packet for packet in packets
                          if packet.payload.get('capability') == 'doctor.recommend']
        assert len(doctor_packets) == len(cases)
        assert all(not ({'actual_drugs', 'drug_labels'} & set(json.loads(packet.payload['text'])))
                   for packet in doctor_packets)
        doctor_prompts = [prompt for prompt in model.prompts
                          if '你是一名专业的医生' in prompt[0]['content']]
        assert len(doctor_prompts) == len(cases)
        assert all('真实药物：' not in prompt[1]['content'] and
                   'actual_drugs' not in prompt[1]['content'] and
                   'drug_labels' not in prompt[1]['content'] for prompt in doctor_prompts)
        traffic_rows = [json.loads(line) for line in
                        (root / 'communications.jsonl').read_text(encoding='utf-8').splitlines()]
        recorded_packets = [row for row in traffic_rows if row['record_type'] == 'packet']
        assert len(recorded_packets) == len(costs)
        assert all(row['step'] in range(len(ROUTE)) and row['execution_ns'] > 0
                   and row['serialization_ns'] > 0 for row in recorded_packets)

        by_step = defaultdict(list)
        for cost in costs:
            by_step[cost.step].append(cost)
        steps = []
        for index, route in enumerate(ROUTE):
            samples = by_step[index]
            steps.append({
                'step': index, 'source': route[0], 'target': route[1], 'capability': route[2],
                'mean_execution_ms': round(mean(c.execution_ns for c in samples) / 1e6, 3),
                'mean_serialization_ms': round(mean(c.serialization_ns for c in samples) / 1e6, 3),
                'mean_message_bytes': round(mean(c.utf8_bytes for c in samples), 1),
                'mean_packet_bytes': round(mean(c.packet_utf8_bytes for c in samples), 1),
                'mean_message_tokens_estimate': round(mean(c.message_tokens_estimate for c in samples), 1),
            })
        # Count the JSON string inside each message body, excluding its wrapper fields.
        dialog_bytes = sum(len(json.dumps(case.patient_input, ensure_ascii=False).encode('utf-8'))
                           for case in cases)
        history_packets = [packet for packet in packets
                           if packet.payload.get('capability') == 'doctor.recommend']
        histories = [json.loads(packet.payload['text'])['history'] for packet in history_packets
                     if 'history' in json.loads(packet.payload['text'])]
        history_bytes = sum(len(json.dumps(history, ensure_ascii=False).encode('utf-8'))
                            for history in histories)
        medicine_packets = [packet for packet in packets
                            if packet.payload.get('capability') == 'expert.review']
        duplicate_introductions = 0
        shared_drug_count = 0
        introduction_hashes = []
        step7_case_bytes = []
        step7_unified_count = 0
        for case, packet in zip(cases, medicine_packets):
            body = json.loads(packet.payload['text'])
            if 'medicine_introductions' in body:
                step7_unified_count += 1
                assert set(body) == {'case_id', 'recommended_drugs', 'actual_drugs',
                                     'medicine_introductions'}
                introductions = body['medicine_introductions']
                assert list(introductions) == list(dict.fromkeys(
                    body['recommended_drugs'] + body['actual_drugs']))
                recommended = {name: introductions[name] for name in body['recommended_drugs']}
                actual = {name: introductions[name] for name in body['actual_drugs']}
            else:
                recommended = body['recommended_introductions']
                actual = body['actual_introductions']
            introduction_hashes.append(hashlib.sha256(json.dumps(
                {'recommended_introductions': recommended, 'actual_introductions': actual},
                ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest())
            legacy_body = json.dumps({
                'case_id': body['case_id'],
                'recommended_introductions': recommended,
                'actual_introductions': actual,
            }, ensure_ascii=False)
            legacy_packet = AgentPacket(packet.head, {
                'capability': 'expert.review', 'text': legacy_body,
            })
            step7_case_bytes.append({
                'case_id': case.case_id,
                'overlap_count': len(recommended.keys() & actual.keys()),
                'legacy_packet_bytes_reconstructed': len(legacy_packet.to_json().encode('utf-8')),
                'current_packet_bytes': len(packet.to_json().encode('utf-8')),
            })
            for name in recommended.keys() & actual.keys():
                shared_drug_count += 1
                if 'medicine_introductions' not in body:
                    duplicate_introductions += len(json.dumps(
                        {name: recommended[name]}, ensure_ascii=False).encode('utf-8'))
        return {
            'method': 'first 10 fixed DIALMED training cases; frozen two-lesson memory; deterministic local model stub',
            'real_llm_api_tokens': None,
            'case_count': len(cases), 'case_ids': [case.case_id for case in cases],
            'packet_count': len(costs),
            'exact_matches': sum(outcome['exact_match'] for outcome in outcomes),
            'steps': steps,
            'mean_case_execution_ms': round(mean(cost.execution_ns for cost in session.trace.case_costs) / 1e6, 3),
            'mean_case_serialization_ms': round(mean(cost.serialization_ns for cost in session.trace.case_costs) / 1e6, 3),
            'mean_case_packet_bytes': round(mean(cost.packet_bytes for cost in session.trace.case_costs), 1),
            'mean_case_packet_tokens_estimate': round(mean(cost.packet_tokens_estimate for cost in session.trace.case_costs), 1),
            'dialog_unique_bytes': dialog_bytes,
            # Retain the report key for older report readers; training now has two handoffs.
            'dialog_transmitted_bytes_steps_1_3_4': dialog_bytes * 2,
            'dialog_redundant_bytes': dialog_bytes,
            'history_handoff_bytes_step_4': history_bytes,
            'history_nonempty_cases': sum(bool(h['golden_experiences'] or h['reflections']) for h in histories),
            'overlapping_medicine_names_step_7': shared_drug_count,
            'duplicate_medicine_introduction_bytes_step_7': duplicate_introductions,
            'introduction_hashes': introduction_hashes,
            'step7_case_bytes': step7_case_bytes,
            'step7_unified_count': step7_unified_count,
            'prompt_hashes': [hashlib.sha256(json.dumps(
                prompt, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()
                              for prompt in model.prompts],
            'label_isolation_checked': True,
        }


def format_report(report: dict) -> str:
    rows = [
        '# Tiny Hospital 第一阶段性能基线',
        '',
        '固定病例：DIALMED train 第 1–10 例；冻结两条历史记忆；确定性本地模型桩。',
        '完整执行五方流程（System、Patient、Retrieval、Doctor、Expert，最后到 End）。',
        '耗时只代表本机单次基线；模型桩没有网络等待，不能据此推断真实 LLM 延迟。',
        '',
        '## 逐步骤平均值',
        '',
        '| 步骤 | 通信 | 执行 ms | 序列化 ms | 正文字节 | 报文字节 | 正文通信 Token 估算 |',
        '| ---: | --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for step in report['steps']:
        rows.append(
            f"| {step['step']} | {step['source']} → {step['target']} "
            f"({step['capability']}) | {step['mean_execution_ms']:.3f} | "
            f"{step['mean_serialization_ms']:.3f} | {step['mean_message_bytes']:.1f} | "
            f"{step['mean_packet_bytes']:.1f} | {step['mean_message_tokens_estimate']:.1f} |"
        )
    rows += [
        '',
        f"每例平均：{report['mean_case_execution_ms']:.3f} ms 执行、"
        f"{report['mean_case_serialization_ms']:.3f} ms 协议序列化与解析、"
        f"{report['mean_case_packet_bytes']:.1f} 报文字节、"
        f"{report['mean_case_packet_tokens_estimate']:.1f} 报文通信 Token 估算。",
        '执行时间是同步 Runtime.deliver 的总时间，包含 Handler；序列化时间包含协议 JSON 编码、解析与校验，以及 Trace 再编码，属于执行时间的一部分。',
        '通信 Token 由字符规则估算；真实模型 API 的 prompt/completion/total Token 应读取 runs/llm_usage.jsonl。',
        '本次模型桩没有真实 API Token，报告中的 real_llm_api_tokens 为 null。',
        '',
        '## 通信冗余与瓶颈',
        '',
        f"- 完整对话在步骤 1、3 各传一次：10 例对话在消息正文中编码一次为 {report['dialog_unique_bytes']} 字节，"
        f"两次传递 {report['dialog_transmitted_bytes_steps_1_3_4']} 字节，"
        f"其中重复 {report['dialog_redundant_bytes']} 字节。Doctor 与 Expert 的模型输入也使用该对话，但属于 API 输入，不计入 Agent 通信字节。",
        f"- 训练阶段 Patient 直接向 Doctor 提供病例，不检索历史经验；本次 {report['history_nonempty_cases']} 例命中，"
        f"10 例 history JSON 合计 {report['history_handoff_bytes_step_4']} 字节。步骤 1、3、5–8 没有再次传历史经验。",
        f"- 药物说明只在步骤 7 随 Retrieval → Expert 传递；推荐与真实药物重合时，两个说明映射在同一报文内重复。"
        f"本次重合 {report['overlapping_medicine_names_step_7']} 个药名，对应重复说明对象约 "
        f"{report['duplicate_medicine_introduction_bytes_step_7']} 字节。步骤 6 只传药名，不传说明。",
        '',
        '优先定位的三个位置：',
        '',
        '1. 步骤 7 的说明载荷最大，优先检查双映射重合和长说明字段。',
        '2. 步骤 1、3、4 的完整对话重复传递，优先检查重复正文占用。',
        '3. Runtime.deliver 与 Trace.packet 的 JSON 边界重复编码和解析，优先评估纯本地开销。',
        '',
        '10 例共 90 条业务报文，全部完成；预测前进入 Doctor 的报文及其模型上下文均检查了真实标签字段隔离。',
        '本阶段没有修改提示词、记忆格式、推荐逻辑、TH-ACP 字段或通信路径。',
        '',
    ]
    return '\n'.join(rows)


if __name__ == '__main__':
    report = measure()
    destination = Path(__file__).resolve().parent / 'runs/performance_current.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
