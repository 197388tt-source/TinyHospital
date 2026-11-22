"""Score completed predictions without calling any model API.

JSON: progress.json with a results array. JSONL: one prediction per line.
Drug lists are treated as sets. Sample averages are not per-drug macro scores.
Failed rows (missing/null predictions) are excluded from quality scores and
reported separately. Unrecorded cases cannot be inferred from progress alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DEFAULT_INPUT = Path(__file__).resolve().parent / 'qwen3_betterex_full2/test/progress.json'


def load_results(path: Path) -> list[dict]:
    text = path.read_text(encoding='utf-8')
    if path.suffix.lower() == '.jsonl':
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        value = json.loads(text)
        rows = value.get('results') if isinstance(value, dict) else value
        if isinstance(value, dict) and 'completed_case_ids' in value:
            if not isinstance(rows, list) or value['completed_case_ids'] != [
                row.get('case_id') if isinstance(row, dict) else None for row in rows
            ]:
                raise ValueError('completed_case_ids 与 results 的病例编号不一致')
    if not isinstance(rows, list) or not rows:
        raise ValueError('结果必须是非空的逐病例数组')
    return rows


def evaluate(rows: list[dict]) -> dict:
    seen = set()
    tp = fp = fn = exact = evaluated = failed = duplicates = 0
    sums = dict(precision=0.0, recall=0.0, f1=0.0, jaccard=0.0)
    failed_ids = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f'第 {index} 条记录不是对象')
        case_id = row.get('case_id')
        if type(case_id) is not int or case_id < 1 or case_id in seen:
            raise ValueError(f'第 {index} 条 case_id 无效或重复')
        seen.add(case_id)
        actual = row.get('actual_drugs', row.get('actual_label'))
        predicted = row.get('recommended_drugs', row.get('predicted_label'))
        for field, names in [('actual', actual), ('predicted', predicted)]:
            if field == 'predicted' and names is None:
                continue
            if not isinstance(names, list) or any(
                not isinstance(name, str) or not name.strip() for name in names
            ):
                raise ValueError(f'病例 {case_id} 的 {field} 必须是非空药名组成的数组')
        if predicted is None:
            failed += 1
            failed_ids.append(case_id)
            continue
        gold, pred = set(actual), set(predicted)
        duplicates += int(len(gold) != len(actual) or len(pred) != len(predicted))
        hit = len(gold & pred)
        tp += hit
        fp += len(pred - gold)
        fn += len(gold - pred)
        evaluated += 1
        exact += pred == gold
        sums['precision'] += hit / len(pred) if pred else 0.0
        sums['recall'] += hit / len(gold) if gold else 0.0
        sums['f1'] += 2 * hit / (len(pred) + len(gold)) if pred or gold else 0.0
        sums['jaccard'] += hit / len(pred | gold) if pred or gold else 0.0
    total = len(rows)
    return {
        'cases': total, 'evaluated': evaluated, 'failed': failed,
        'failed_case_ids': failed_ids, 'failure_rate': failed / total if total else 0.0,
        'exact_matches': exact,
        'exact_match_accuracy': exact / evaluated if evaluated else 0.0,
        'exact_match_accuracy_all_recorded': exact / total if total else 0.0,
        'micro_precision': tp / (tp + fp) if tp + fp else 0.0,
        'micro_recall': tp / (tp + fn) if tp + fn else 0.0,
        'micro_f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        'micro_jaccard': tp / (tp + fp + fn) if tp + fp + fn else 0.0,
        **{'sample_average_' + key: value / evaluated if evaluated else 0.0
           for key, value in sums.items()},
        'true_positive': tp, 'false_positive': fp, 'false_negative': fn,
        'cases_with_duplicate_drugs': duplicates,
        'scoring_policy': {
            'drug_comparison': 'exact names, sets; duplicates removed',
            'failed_predictions': 'missing/null excluded from quality scores; empty list is a valid prediction',
            'zero_denominator': '0; two empty sets still count as exact match',
            'coverage': 'recorded cases only; unfinished/unrecorded cases not inferred',
            'sample_average': 'mean of per-case scores, not per-label macro scores',
        },
    }


def main():
    parser = argparse.ArgumentParser(description='计算 progress.json / JSONL 药物推荐结果指标（不调用 API）')
    parser.add_argument('--input', '--predictions', dest='input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path, help='默认保存到输入文件同目录 metrics.json')
    parser.add_argument('--expected-count', type=int, help='预期病例数，用于检查记录数量，不补造缺失预测')
    args = parser.parse_args()
    if args.expected_count is not None and args.expected_count < 1:
        parser.error('--expected-count 必须为正整数')
    output = args.output or args.input.parent / 'metrics.json'
    if output.resolve() == args.input.resolve():
        parser.error('输出路径不能与输入路径相同')
    try:
        metrics = evaluate(load_results(args.input))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    metrics['source'] = str(args.input.resolve())
    metrics['source_sha256'] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    if args.expected_count is not None:
        metrics['expected_cases'] = args.expected_count
        metrics['recorded_count_matches_expected'] = metrics['cases'] == args.expected_count
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f"记录 {metrics['cases']} 例；参与评分 {metrics['evaluated']} 例；失败 {metrics['failed']} 例")
    for key in ('micro_precision', 'micro_recall', 'micro_f1', 'micro_jaccard',
                'exact_match_accuracy', 'sample_average_f1', 'sample_average_jaccard'):
        print(f'{key}: {metrics[key]:.6f} ({metrics[key]:.2%})')
    if args.expected_count is not None and metrics['cases'] != args.expected_count:
        print(f'注意：记录数量与预期 {args.expected_count} 例不一致，不能作为完整测试集成绩。')
    print(f'指标已保存：{output}')


if __name__ == '__main__':
    main()
