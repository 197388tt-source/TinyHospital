"""Build paired traincut experience-ablation tables from completed runs."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
EXPERIMENT = ROOT / 'experiment'
RUNS = {
    'empty': 'qwen_traincut_glm3_gpu0_empty',
    'positive': 'qwen_traincut_glm3_gpu0_positive_only',
    'negative': 'qwen_traincut_glm3_gpu0_reflection_only',
    'both': 'qwen_traincut_glm3_gpu0_leave_one_out',
}
KINDS = {'positive': 'golden_experiences', 'negative': 'reflections'}
LABELS = {'positive': '正向', 'negative': '反向'}


def read_run(name):
    folder = EXPERIMENT / RUNS[name] / 'test'
    state = json.loads((folder / 'progress.json').read_text(encoding='utf-8'))
    results = {row['case_id']: row for row in state['results']}
    assert len(results) == len(state['results']) == 686, (name, len(results))
    retrieval = {}
    with (folder / 'diagnostics.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            event = json.loads(line)
            if event.get('event') == 'retrieval':
                case_id = event['case_id']
                assert case_id not in retrieval, (name, case_id)
                selected = {}
                for candidate in event['candidates']:
                    if candidate['reason'] != 'selected':
                        continue
                    kind = candidate['kind']
                    assert kind not in selected, (name, case_id, kind)
                    assert candidate['source_case_id'] != case_id, (name, case_id)
                    selected[kind] = candidate
                retrieval[case_id] = selected
    assert results.keys() == retrieval.keys(), name
    return results, retrieval


def case_f1(row):
    prediction = Counter(row.get('recommended_drugs') or [])
    target = Counter(row.get('actual_drugs') or [])
    tp = sum((prediction & target).values())
    fp = sum((prediction - target).values())
    fn = sum((target - prediction).values())
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0


def transition(empty, treatment):
    before = bool(empty['exact_match'])
    after = bool(treatment['exact_match'])
    return {
        'success': int(after),
        'failure': int(not after),
        'gain': int(not before and after),
        'harm': int(before and not after),
        'both_correct': int(before and after),
        'both_wrong': int(not before and not after),
        'f1_delta': case_f1(treatment) - case_f1(empty),
        'doctor_failure': int(treatment.get('status') == 'doctor_failed'),
    }


def fmt_drugs(row):
    return '、'.join(row.get('recommended_drugs') or [])


def write_csv(name, headers, rows):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def summarize_hits(case_ids, results, empty_results):
    stats = Counter()
    changes = []
    for case_id in case_ids:
        change = transition(empty_results[case_id], results[case_id])
        for key in ('success', 'failure', 'gain', 'harm', 'both_correct',
                    'both_wrong', 'doctor_failure'):
            stats[key] += change[key]
        delta = change['f1_delta']
        stats['f1_up' if delta > 1e-12 else
              'f1_down' if delta < -1e-12 else 'f1_same'] += 1
        changes.append(delta)
    stats['hits'] = len(case_ids)
    return stats, sum(changes) / len(changes) if changes else None


def metrics(results):
    tp = fp = fn = 0
    for row in results.values():
        predicted = Counter(row.get('recommended_drugs') or [])
        actual = Counter(row.get('actual_drugs') or [])
        tp += sum((predicted & actual).values())
        fp += sum((predicted - actual).values())
        fn += sum((actual - predicted).values())
    return {
        'cases': len(results), 'tp': tp, 'fp': fp, 'fn': fn,
        'micro_precision': tp / (tp + fp),
        'micro_recall': tp / (tp + fn),
        'micro_f1': 2 * tp / (2 * tp + fp + fn),
        'micro_jaccard': tp / (tp + fp + fn),
        'exact_match': sum(bool(row['exact_match']) for row in results.values()),
        'doctor_failures': sum(row.get('status') == 'doctor_failed'
                               for row in results.values()),
    }


def main():
    data = {name: read_run(name) for name in RUNS}
    empty = data['empty'][0]
    case_ids = sorted(empty)
    assert all(set(results) == set(case_ids) for results, _ in data.values())
    assert all(data['empty'][1][case_id] == {} for case_id in case_ids)
    assert all(data['positive'][1][case_id].keys() <= {KINDS['positive']}
               for case_id in case_ids)
    assert all(data['negative'][1][case_id].keys() <= {KINDS['negative']}
               for case_id in case_ids)
    for case_id in case_ids:
        truth = empty[case_id]['actual_drugs']
        assert all(results[case_id]['actual_drugs'] == truth
                   for results, _ in data.values()), case_id

    memory_dir = EXPERIMENT / 'qwen_train_deepseek_full'
    experiences = {}
    for name, filename, field in (
        ('positive', 'golden_experience.json', 'experience'),
        ('negative', 'reflection_memory.json', 'reflection'),
    ):
        records = json.loads((memory_dir / filename).read_text(encoding='utf-8'))
        for record in records:
            key = (name, record['case_id'])
            assert key not in experiences
            experiences[key] = record[field]
    assert len(experiences) == 559

    single_rows = []
    both_rows = []
    single_hits = defaultdict(list)
    both_hits = defaultdict(list)
    for case_id in case_ids:
        baseline = empty[case_id]
        for name in ('positive', 'negative'):
            results, retrieval = data[name]
            treatment = results[case_id]
            selected = retrieval[case_id].get(KINDS[name])
            change = transition(baseline, treatment)
            source_id = selected['source_case_id'] if selected else None
            if source_id is not None:
                assert (name, source_id) in experiences
                single_hits[(name, source_id)].append(case_id)
            single_rows.append({
                '经验类型': LABELS[name], '经验来源病例ID': source_id,
                '测试病例ID': case_id,
                '相似度': selected['similarity'] if selected else '',
                '经验文本': experiences[(name, source_id)] if selected else '',
                '真实药物': '、'.join(baseline['actual_drugs']),
                '空经验预测': fmt_drugs(baseline),
                '单类经验预测': fmt_drugs(treatment),
                '空经验完全正确': int(bool(baseline['exact_match'])),
                '单类经验完全正确': change['success'],
                '空经验错→单类对': change['gain'],
                '空经验对→单类错': change['harm'],
                '空经验病例F1': round(case_f1(baseline), 6),
                '单类经验病例F1': round(case_f1(treatment), 6),
                '病例F1变化': round(change['f1_delta'], 6),
                '单类实验医生失败': change['doctor_failure'],
            })

        results, retrieval = data['both']
        treatment = results[case_id]
        change = transition(baseline, treatment)
        selected = retrieval[case_id]
        found = {}
        for name in ('positive', 'negative'):
            candidate = selected.get(KINDS[name])
            found[name] = candidate
            if candidate:
                source_id = candidate['source_case_id']
                assert (name, source_id) in experiences
                both_hits[(name, source_id)].append(case_id)
        both_rows.append({
            '测试病例ID': case_id,
            '正向经验来源病例ID': found['positive']['source_case_id'] if found['positive'] else '',
            '正向相似度': found['positive']['similarity'] if found['positive'] else '',
            '正向经验文本': experiences[('positive', found['positive']['source_case_id'])]
                         if found['positive'] else '',
            '反向经验来源病例ID': found['negative']['source_case_id'] if found['negative'] else '',
            '反向相似度': found['negative']['similarity'] if found['negative'] else '',
            '反向经验文本': experiences[('negative', found['negative']['source_case_id'])]
                         if found['negative'] else '',
            '真实药物': '、'.join(baseline['actual_drugs']),
            '空经验预测': fmt_drugs(baseline),
            '正反都有预测': fmt_drugs(treatment),
            '空经验完全正确': int(bool(baseline['exact_match'])),
            '正反都有完全正确': change['success'],
            '空经验错→正反对': change['gain'],
            '空经验对→正反错': change['harm'],
            '空经验病例F1': round(case_f1(baseline), 6),
            '正反都有病例F1': round(case_f1(treatment), 6),
            '病例F1变化': round(change['f1_delta'], 6),
            '正反都有医生失败': change['doctor_failure'],
        })

    write_csv('per_case.csv', list(single_rows[0]), single_rows)
    write_csv('per_case_both.csv', list(both_rows[0]), both_rows)

    single_aggregate = []
    both_aggregate = []
    for (name, source_id), text in sorted(experiences.items(),
                                          key=lambda item: (0 if item[0][0] == 'positive' else 1,
                                                            item[0][1])):
        for output, hit_map, run_name, count_label, ids_label, qualifier in (
            (single_aggregate, single_hits, name, '单类命中次数', '单类命中病例ID', '单类'),
            (both_aggregate, both_hits, 'both', '正反都有命中次数',
             '正反都有命中病例ID', '正反都有'),
        ):
            hit_ids = hit_map[(name, source_id)]
            stats, average = summarize_hits(hit_ids, data[run_name][0], empty)
            output.append({
                '经验类型': LABELS[name], '经验来源病例ID': source_id,
                '经验文本': text, count_label: stats['hits'],
                f'{qualifier}预测成功次数': stats['success'],
                f'{qualifier}预测失败次数': stats['failure'],
                f'空经验错→{qualifier}对': stats['gain'],
                f'空经验对→{qualifier}错': stats['harm'],
                '完全正确净变化': stats['gain'] - stats['harm'],
                '两者都对': stats['both_correct'],
                '两者都错': stats['both_wrong'],
                '病例F1提高次数': stats['f1_up'],
                '病例F1下降次数': stats['f1_down'],
                '病例F1净提高次数': stats['f1_up'] - stats['f1_down'],
                '病例F1不变次数': stats['f1_same'],
                '命中病例平均F1变化': round(average, 6) if average is not None else '',
                f'{qualifier}医生失败次数': stats['doctor_failure'],
                ids_label: '、'.join(map(str, hit_ids)),
            })
    assert len(single_aggregate) == len(both_aggregate) == 559
    write_csv('per_experience.csv', list(single_aggregate[0]), single_aggregate)
    write_csv('per_experience_both.csv', list(both_aggregate[0]), both_aggregate)

    summary = {
        'dataset': 'data/traincut.txt',
        'cases': len(case_ids),
        'experience_records': {'positive': 59, 'negative': 500},
        'runs': {name: metrics(results) for name, (results, _) in data.items()},
        'comparisons_to_empty': {},
        'retrieval': {},
    }
    for name in ('positive', 'negative', 'both'):
        results, retrieval = data[name]
        stats, average = summarize_hits(case_ids, results, empty)
        summary['comparisons_to_empty'][name] = {
            **{key: value for key, value in stats.items() if key != 'hits'},
            'paired_cases': stats['hits'],
            'mean_case_f1_delta': round(average, 6)}
        selected = data[name][1]
        summary['retrieval'][name] = {
            'positive_hits': sum(KINDS['positive'] in retrieval[c] for c in case_ids),
            'negative_hits': sum(KINDS['negative'] in retrieval[c] for c in case_ids),
            'unique_positive_experiences': len({
                selected[c][KINDS['positive']]['source_case_id'] for c in case_ids
                if KINDS['positive'] in selected[c]}),
            'unique_negative_experiences': len({
                selected[c][KINDS['negative']]['source_case_id'] for c in case_ids
                if KINDS['negative'] in selected[c]}),
        }
    summary['retrieval']['both']['same_source_as_single'] = {
        name: sum(
            data[name][1][case_id].get(KINDS[name], {}).get('source_case_id') ==
            data['both'][1][case_id].get(KINDS[name], {}).get('source_case_id')
            for case_id in case_ids)
        for name in ('positive', 'negative')
    }
    (OUT / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
