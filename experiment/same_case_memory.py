"""Paired train-set evaluation using only each case's own saved lessons.

Run from tiny_hospital:
    python experiment/same_case_memory.py --dry-run
    CUDA_VISIBLE_DEVICES=0 python experiment/same_case_memory.py
    python experiment/same_case_memory.py --resume

This measures reuse on the source cases, not generalization to unseen cases.
No embedding, expert calls, or memory updates are performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import fcntl
import hashlib
import json
import math
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.doctor_Qwen import DoctorAgent
from agent.validation import _json_object
from cases import DIALMED_TRAIN_PATH, load_dialmed_cases
from experiment.test import evaluate
from tool.experiment_progress import atomic_json
from tool.local_qwen_client import DEFAULT_MODEL_PATH, LocalQwenClient
from tool.medicine_store import MedicineKnowledgeBase
from tool.tracing import Trace


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (ValueError, OSError) as exc:
        raise ValueError(f'无法读取 JSON：{path}：{exc}') from exc


def load_memories(directory: Path, case_ids: set[int], kind: str) -> tuple[list, dict]:
    """IDs belong to dataset lines; never zip memory rows with dataset rows."""
    snapshot, lookup = [], {}
    for filename, field, bucket in (
        ('golden_experience.json', 'experience', 'golden_experiences'),
        ('reflection_memory.json', 'reflection', 'reflections'),
    ):
        rows = read_json(directory / filename)
        if not isinstance(rows, list):
            raise ValueError(f'{filename} 必须为 JSON 数组；无经验时填写 []')
        seen = set()
        for row in rows:
            if (not isinstance(row, dict) or type(row.get('case_id')) is not int
                    or row['case_id'] not in case_ids or row['case_id'] in seen
                    or not isinstance(row.get(field), str) or not row[field].strip()):
                raise ValueError(f'{filename} 存在无效、重复或不属于训练集的 case_id/经验')
            seen.add(row['case_id'])
            if kind == 'golden' and bucket == 'reflections':
                continue
            if kind == 'reflection' and bucket == 'golden_experiences':
                continue
            history = lookup.setdefault(row['case_id'], empty_history())
            history[bucket].append(row[field])
        snapshot.append(rows)
    return snapshot, lookup


def empty_history() -> dict:
    return {'golden_experiences': [], 'reflections': []}


def memory_type(history: dict) -> str:
    golden, reflection = bool(history['golden_experiences']), bool(history['reflections'])
    return {(False, False): 'none', (True, False): 'golden',
            (False, True): 'reflection', (True, True): 'both'}[(golden, reflection)]


class RecordedDoctor(DoctorAgent):
    """Keep recommend(), its test prompts, validation and retries unchanged.

    Only replace the transport logging hook: this script has no packet Runtime.
    """

    def ask_llm(self, step: str, instruction: str, context: str) -> str:
        messages = [{'role': 'system', 'content': instruction},
                    {'role': 'user', 'content': context}]
        started = perf_counter()
        reply = self.llm.chat(messages)
        usage = getattr(self.llm, 'last_usage', None)
        self.model_calls.append({
            'step': step, 'messages': messages, 'raw_reply': reply,
            'elapsed_seconds': perf_counter() - started,
            'usage': {key: getattr(usage, key, None)
                      for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')},
        })
        return reply


def predict(case, history: dict, client, medicines) -> dict:
    doctor = RecordedDoctor(Trace(verbose=False), medicines, lambda: client, training=False)
    # Only the known target count is provided, exactly as in the existing test.
    # Labels are attached after recommend() returns, never passed to the model.
    result = json.loads(doctor.recommend(json.dumps({
        'case_id': case.case_id, 'count': case.drug_count,
        'patient_dialog': case.patient_input, 'history': history,
    }, ensure_ascii=False)))
    result['actual_drugs'] = list(case.drug_labels)
    result['exact_match'] = (result['recommended_drugs'] is not None
                             and set(result['recommended_drugs']) == set(case.drug_labels))
    result['model_calls'] = doctor.model_calls
    if result['recommended_drugs'] is not None:
        result['reasoning_summary'] = _json_object(
            doctor.model_calls[-1]['raw_reply'])['reasoning_summary']
    return result


def comparison(rows: list[dict]) -> dict:
    transitions = {'wrong_to_right': [], 'right_to_wrong': [],
                   'both_right': [], 'both_wrong': []}
    for row in rows:
        before, after = row['baseline']['exact_match'], row['exact_match']
        key = {(False, True): 'wrong_to_right', (True, False): 'right_to_wrong',
               (True, True): 'both_right', (False, False): 'both_wrong'}[(before, after)]
        transitions[key].append(row['case_id'])
    paired = [r for r in rows if r['recommended_drugs'] is not None
              and r['baseline']['recommended_drugs'] is not None]
    return {
        'cases': len(rows),
        'baseline': evaluate([r['baseline'] for r in rows]),
        'with_memory': evaluate(rows),
        'paired_valid_cases': len(paired),
        'paired_valid_baseline': evaluate([r['baseline'] for r in paired]),
        'paired_valid_with_memory': evaluate(paired),
        'transition_policy': '生成失败计为不正确；失败数在各组指标中单独报告',
        'transitions': transitions,
        'net_correct_gain': len(transitions['wrong_to_right']) - len(transitions['right_to_wrong']),
    }


def save_reports(directory: Path, state: dict) -> dict:
    rows = state['results']
    covered = [r for r in rows if r['memory_type'] != 'none']
    metrics = {
        'experiment': 'same_case_memory',
        'interpretation': '训练原病例经验回放；不能作为未见病例泛化成绩',
        'expected_cases': len(state['config']['case_ids']),
        'completed_cases': len(rows),
        'complete': len(rows) == len(state['config']['case_ids']),
        'all_cases': comparison(rows),
        'cases_with_memory': comparison(covered),
        'by_memory_type': {kind: comparison([r for r in rows if r['memory_type'] == kind])
                           for kind in ('golden', 'reflection', 'both', 'none')},
    }
    atomic_json(directory / 'metrics.json', metrics)
    atomic_json(directory / 'baseline_progress.json', {
        'config': state['config'], 'completed_case_ids': state['completed_case_ids'],
        'results': [r['baseline'] for r in rows],
    })
    with (directory / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['case_id', '经验类型', '标准答案', '无经验预测', '同ID经验预测',
                         '无经验正确', '同ID经验正确', '正向经验', '反思'])
        for r in rows:
            show = lambda drugs: '、'.join(drugs) if drugs is not None else '[生成失败]'
            writer.writerow([r['case_id'], r['memory_type'], show(r['actual_drugs']),
                             show(r['baseline']['recommended_drugs']), show(r['recommended_drugs']),
                             r['baseline']['exact_match'], r['exact_match'],
                             '\n'.join(r['history']['golden_experiences']),
                             '\n'.join(r['history']['reflections'])])
    return metrics


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', type=Path, default=DIALMED_TRAIN_PATH)
    parser.add_argument('--memory-dir', type=Path, default=ROOT / 'experiment/qwen_train_deepseek_test')
    parser.add_argument('--experiment-dir', type=Path, default=ROOT / 'experiment/qwen_same_case_memory')
    parser.add_argument('--memory-kind', choices=('both', 'golden', 'reflection'), default='both')
    parser.add_argument('--count', type=int, help='只运行训练集前 N 条，保留原始病例编号')
    parser.add_argument('--doctor-model-path', type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument('--doctor-max-new-tokens', type=int, default=1024)
    parser.add_argument('--chat-timeout', type=float, default=120)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='检查数据和经验匹配，不加载模型或写结果')
    args = parser.parse_args(argv)
    if args.count is not None and args.count < 1:
        parser.error('--count 必须为正整数')
    if args.doctor_max_new_tokens < 1 or not math.isfinite(args.chat_timeout) or args.chat_timeout <= 0:
        parser.error('生成长度和超时时间必须为正数')
    for key in ('dataset', 'memory_dir', 'experiment_dir', 'doctor_model_path'):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    if args.experiment_dir == args.memory_dir:
        parser.error('输出目录必须与经验来源目录不同')
    try:
        cases = load_dialmed_cases(args.dataset)
        snapshot, memories = load_memories(args.memory_dir, {c.case_id for c in cases}, args.memory_kind)
        # When available, verify that IDs refer to the dataset that generated the memories.
        source_progress = args.memory_dir / 'train/progress.json'
        if source_progress.exists():
            expected = read_json(source_progress).get('config', {}).get('dataset_sha256')
            if expected and expected != sha256(args.dataset):
                raise ValueError('训练集与经验来源 train/progress.json 的数据哈希不一致')
        if args.count is not None:
            if args.count > len(cases):
                raise ValueError('--count 超出训练集病例数量')
            cases = cases[:args.count]
        medicines = MedicineKnowledgeBase()
        for case in cases:
            if not set(case.drug_labels) <= set(medicines.names):
                raise ValueError(f'病例 {case.case_id} 的标准答案含白名单外药物')
        counts = Counter(memory_type(memories.get(c.case_id, empty_history())) for c in cases)
        print(f'病例 {len(cases)}；经验匹配：{dict(counts)}', flush=True)
        print('读取经验文件：' + str(args.memory_dir), flush=True)
        if args.dry_run:
            return
        if counts['none'] == len(cases):
            raise ValueError('所选病例没有对应经验，无法验证经验作用；请检查经验文件或 --memory-kind')
        files = ['experiment/same_case_memory.py', 'agent/doctor_Qwen.py', 'agent/base.py',
                 'agent/validation.py', 'cases.py', 'experiment/test.py', 'tool/case_utils.py',
                 'tool/local_qwen_client.py', 'tool/medicine_store.py', 'tool/experiment_progress.py']
        config = {
            'dataset': str(args.dataset), 'dataset_sha256': sha256(args.dataset),
            'memory_dir': str(args.memory_dir), 'memory_kind': args.memory_kind,
            'memory_sha256': hashlib.sha256(json.dumps(snapshot, ensure_ascii=False,
                                                       sort_keys=True).encode()).hexdigest(),
            'doctor_model_path': str(args.doctor_model_path),
            'doctor_generation': {'enable_thinking': False, 'do_sample': False,
                                  'max_new_tokens': args.doctor_max_new_tokens},
            'chat_timeout': args.chat_timeout, 'case_ids': [c.case_id for c in cases],
            'code_sha256': {f: sha256(ROOT / f) for f in files},
            'medicine_sha256': sha256(medicines.catalog_path),
        }
        args.experiment_dir.mkdir(parents=True, exist_ok=True)
        with (args.experiment_dir / '.same_case.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            progress = args.experiment_dir / 'progress.json'
            if args.resume:
                state = read_json(progress)
                if state['config'] != config or state['memory'] != snapshot:
                    raise ValueError('配置、代码、数据或经验已变化，请使用新的输出目录')
                completed = [r['case_id'] for r in state['results']]
                if completed != state['completed_case_ids'] or completed != config['case_ids'][:len(completed)]:
                    raise ValueError('进度病例编号不是当前训练集的连续已完成前缀')
            else:
                if progress.exists():
                    raise ValueError('已有实验结果，请使用 --resume 或新的 --experiment-dir')
                state = {'config': config, 'memory': snapshot, 'completed_case_ids': [], 'results': []}
                atomic_json(progress, state)
            pending = cases[len(state['results']):]
            if pending:
                client = LocalQwenClient(args.doctor_model_path, args.doctor_max_new_tokens,
                                         timeout=args.chat_timeout)
                for case in pending:
                    history = memories.get(case.case_id, empty_history())
                    baseline = predict(case, empty_history(), client, medicines)
                    result = (predict(case, history, client, medicines)
                              if memory_type(history) != 'none' else copy.deepcopy(baseline))
                    result.update(history=history, memory_type=memory_type(history),
                                  baseline=baseline, baseline_reused=memory_type(history) == 'none')
                    state['results'].append(result)
                    state['completed_case_ids'].append(case.case_id)
                    atomic_json(progress, state)
                    save_reports(args.experiment_dir, state)
                    print(f"[{len(state['results'])}/{len(cases)}] 病例 {case.case_id} "
                          f"{result['memory_type']}：无经验 {baseline['exact_match']} → "
                          f"同ID经验 {result['exact_match']}", flush=True)
            metrics = save_reports(args.experiment_dir, state)
            effect = metrics['cases_with_memory']
            print(f"有经验病例 {effect['cases']}；纠正 {len(effect['transitions']['wrong_to_right'])}；"
                  f"退化 {len(effect['transitions']['right_to_wrong'])}；净增 {effect['net_correct_gain']}")
            print(f'结果目录：{args.experiment_dir}')
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
