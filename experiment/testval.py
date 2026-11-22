"""Score devcut predictions using the same metric definitions as test.py."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

if __package__:
    from .test import evaluate, load_results
else:
    from test import evaluate, load_results


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / 'experiment/qwen_validation/test/progress.json'
DEFAULT_DATASET = ROOT / 'data/devcut.txt'


def main():
    parser = argparse.ArgumentParser(description='计算 devcut 验证集药物推荐指标（不调用模型）')
    parser.add_argument('--input', '--predictions', dest='input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--output', type=Path, help='默认保存到输入文件同目录 metrics.json')
    args = parser.parse_args()
    output = args.output or args.input.parent / 'metrics.json'
    if output.resolve() in (args.input.resolve(), args.dataset.resolve()):
        parser.error('输出路径不能覆盖预测文件或数据集')
    try:
        rows = load_results(args.input)
        metrics = evaluate(rows)
        labels = {}
        for line_number, line in enumerate(args.dataset.read_text(encoding='utf-8').splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            actual = record.get('label') if isinstance(record, dict) else None
            if not isinstance(actual, list) or not actual or any(
                not isinstance(name, str) or not name.strip() for name in actual
            ):
                raise ValueError(f'验证集第 {line_number} 行 label 无效')
            labels[line_number] = set(actual)
        if not labels:
            raise ValueError('验证集不能为空')
        for row in rows:
            case_id = row['case_id']
            if case_id not in labels:
                raise ValueError(f'病例 {case_id} 不在验证集中')
            if set(row.get('actual_drugs', row.get('actual_label'))) != labels[case_id]:
                raise ValueError(f'病例 {case_id} 的结果标签与验证集不一致')
        missing = sorted(set(labels) - {row['case_id'] for row in rows})
        metrics.update(
            source=str(args.input.resolve()),
            source_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
            dataset=str(args.dataset.resolve()),
            dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
            expected_cases=len(labels),
            recorded_count_matches_expected=not missing,
            missing_case_ids=missing,
            recorded_coverage=len(rows) / len(labels),
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"验证集 {len(labels)} 例；记录 {metrics['cases']} 例；参与评分 {metrics['evaluated']} 例；失败 {metrics['failed']} 例")
    for key in ('micro_precision', 'micro_recall', 'micro_f1', 'micro_jaccard',
                'exact_match_accuracy', 'exact_match_accuracy_all_recorded',
                'sample_average_precision', 'sample_average_recall',
                'sample_average_f1', 'sample_average_jaccard'):
        print(f'{key}: {metrics[key]:.6f} ({metrics[key]:.2%})')
    if missing:
        print(f'注意：尚缺 {len(missing)} 例，当前指标仅代表已记录结果。')
    print(f'指标已保存：{output}')


if __name__ == '__main__':
    main()
