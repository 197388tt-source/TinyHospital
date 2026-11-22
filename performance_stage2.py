"""Compare the current ten-case run with the frozen first-stage baseline."""

from __future__ import annotations

import json
from pathlib import Path

from performance_baseline import measure


ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / 'runs/performance_baseline.json'
OUTPUT = ROOT / 'runs/performance_stage2.json'


def compare() -> dict:
    baseline = json.loads(BASELINE.read_text(encoding='utf-8'))
    current = measure()
    if baseline['case_ids'] != current['case_ids'] or baseline['packet_count'] != current['packet_count']:
        raise RuntimeError('第一阶段与第二阶段病例或通信步骤不一致')
    if baseline['introduction_hashes'] != current['introduction_hashes']:
        raise RuntimeError('Expert 还原后的药物说明与第一阶段不一致')
    if baseline['prompt_hashes'] != current['prompt_hashes']:
        raise RuntimeError('模型提示词输入与第一阶段不一致')
    if not current['label_isolation_checked']:
        raise RuntimeError('医生预测前标签隔离检查失败')
    if current['step7_unified_count'] != current['case_count']:
        raise RuntimeError('步骤 7 未全部使用统一药物说明映射')
    steps = []
    for old, new in zip(baseline['steps'], current['steps']):
        if old['step'] != new['step'] or old['capability'] != new['capability']:
            raise RuntimeError('第一阶段与第二阶段通信顺序不一致')
        steps.append({
            'step': new['step'], 'route': f"{new['source']} → {new['target']}",
            'baseline_packet_bytes': old['mean_packet_bytes'],
            'stage2_packet_bytes': new['mean_packet_bytes'],
            'packet_bytes_change': round(new['mean_packet_bytes'] - old['mean_packet_bytes'], 1),
            'baseline_serialization_ms': old['mean_serialization_ms'],
            'stage2_serialization_ms': new['mean_serialization_ms'],
            'baseline_execution_ms': old['mean_execution_ms'],
            'stage2_execution_ms': new['mean_execution_ms'],
        })
    return {
        'baseline_file': str(BASELINE.relative_to(ROOT)),
        'method': current['method'],
        'case_count': current['case_count'],
        'packet_count': current['packet_count'],
        'introduction_hashes_match': True,
        'prompt_hashes_match': True,
        'label_isolation_checked': True,
        'real_llm_api_tokens': None,
        'step_comparison': steps,
        'case_comparison': {
            'baseline_packet_bytes': baseline['mean_case_packet_bytes'],
            'stage2_packet_bytes': current['mean_case_packet_bytes'],
            'baseline_serialization_ms': baseline['mean_case_serialization_ms'],
            'stage2_serialization_ms': current['mean_case_serialization_ms'],
            'baseline_execution_ms': baseline['mean_case_execution_ms'],
            'stage2_execution_ms': current['mean_case_execution_ms'],
        },
        'step7_cases': current['step7_case_bytes'],
        'current': current,
    }


def format_report(result: dict) -> str:
    total = result['case_comparison']
    rows = [
        '# Tiny Hospital 第二阶段性能报告',
        '',
        '固定 DIALMED train 第 1–10 例，冻结同两条历史记忆，使用第一阶段相同的确定性模型桩。',
        '第一阶段数据来自 runs/performance_baseline.json；第二阶段为本次单次运行。',
        '模型桩没有真实 API 等待或实际 API Token；耗时变化受本机运行波动影响，字节差异为确定值。',
        '',
        '## 步骤 7 的变更',
        '',
        '原报文分别包含 recommended_introductions 与 actual_introductions，重合药名的说明出现两次。',
        '现报文包含 recommended_drugs、actual_drugs 和 medicine_introductions；每个药名的说明只出现一次。',
        'Expert 按原药名顺序重建两份本地映射。10 例还原映射和全部 Doctor/Expert 模型提示词的摘要均与第一阶段逐例一致。',
        '',
        '## 逐步骤比较（每例平均）',
        '',
        '| 步骤 | 通信 | 报文字节 基线→二阶段 | 变化字节 | 序列化 ms 基线→二阶段 | 执行 ms 基线→二阶段 |',
        '| ---: | --- | ---: | ---: | ---: | ---: |',
    ]
    for step in result['step_comparison']:
        rows.append(
            f"| {step['step']} | {step['route']} | "
            f"{step['baseline_packet_bytes']:.1f} → {step['stage2_packet_bytes']:.1f} | "
            f"{step['packet_bytes_change']:+.1f} | "
            f"{step['baseline_serialization_ms']:.3f} → {step['stage2_serialization_ms']:.3f} | "
            f"{step['baseline_execution_ms']:.3f} → {step['stage2_execution_ms']:.3f} |"
        )
    rows += [
        '',
        f"全流程每例：报文 {total['baseline_packet_bytes']:.1f} → {total['stage2_packet_bytes']:.1f} 字节；"
        f"序列化 {total['baseline_serialization_ms']:.3f} → {total['stage2_serialization_ms']:.3f} ms；"
        f"执行 {total['baseline_execution_ms']:.3f} → {total['stage2_execution_ms']:.3f} ms。",
        '',
        '## 步骤 7 分病例报文字节',
        '',
        '| 病例 | 重合药名数 | 原结构重建字节 | 二阶段字节 | 节省字节 |',
        '| ---: | ---: | ---: | ---: | ---: |',
    ]
    for item in result['step7_cases']:
        saving = item['legacy_packet_bytes_reconstructed'] - item['current_packet_bytes']
        rows.append(
            f"| {item['case_id']} | {item['overlap_count']} | "
            f"{item['legacy_packet_bytes_reconstructed']} | {item['current_packet_bytes']} | {saving:+d} |"
        )
    rows += [
        '',
        '没有重合药名时，两份药名列表带来少量额外字节；有重合时，复用药物说明抵消该开销。',
        '原结构分病例字节由相同 TH-ACP 报文头与第一阶段说明映射重建；第一阶段总量取冻结基线文件。',
        '完整对话、历史检索、TH-ACP 字段及通信顺序均未改变。',
        '10 例共 90 条业务报文，预测前 Doctor 输入的真实标签字段隔离检查通过。',
        '',
    ]
    return '\n'.join(rows)


if __name__ == '__main__':
    report = compare()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    OUTPUT.with_suffix('.md').write_text(format_report(report), encoding='utf-8')
    print(json.dumps(report['case_comparison'], ensure_ascii=False, indent=2))
