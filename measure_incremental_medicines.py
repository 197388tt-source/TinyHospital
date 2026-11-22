"""Measure the optional retrieval.medicines reply delta with real Pipe frames."""

from __future__ import annotations

import json
import multiprocessing as mp
import tempfile
from pathlib import Path
from statistics import mean, median

from cases import DIALMED_TRAIN_PATH, load_dialmed_cases
from hospital_process import ProcessHospitalSession
from performance_baseline import FixedModel


OUTPUT = Path(__file__).resolve().parent / 'runs/stage8_incremental_medicines.json'


def _paths(root: Path) -> dict:
    return {
        'golden_path': root / 'golden.json',
        'reflection_path': root / 'reflection.json',
        'communication_path': root / 'communication.jsonl',
        'control_path': root / 'control.jsonl',
        'llm_path': root / 'llm.jsonl',
        'dataset_path': DIALMED_TRAIN_PATH,
    }


def _timings(values: list[int]) -> dict:
    return {'samples': len(values), 'mean_us': round(mean(values) / 1000, 2),
            'median_us': round(median(values) / 1000, 2),
            'total_us': round(sum(values) / 1000, 2)}


def _trial(root: Path, cases: list, incremental: bool):
    with ProcessHospitalSession(
        retrieval_mode='jaccard', verbose=False, training=True, llm_client_factory=FixedModel,
        shared_dialog=False, incremental_history=False,
        incremental_medicines=incremental, **_paths(root),
    ) as session:
        outcomes = [session.run_case(case) for case in cases]
        frames = session.runtime.transport_frame_stats
        selected = [row for row in frames if row['capability'] == 'retrieval.medicines']
        reply_costs = [row for row in session.trace.packet_costs
                       if row.capability == 'retrieval.medicines'
                       and row.source == 'RetrievalAgent']
        expert_inputs = [packet.payload['text'] for packet in session.trace.packets
                         if packet.payload.get('capability') == 'expert.review'
                         and packet.head.source == 'RetrievalAgent']
        child_encoding = session.runtime.control_agent(
            'RetrievalAgent', 'snapshot')['medicine_response_encoding']
        extra = session.medicine_delta_stats
        assert len(selected) == len(reply_costs) == len(child_encoding) == len(cases)
        assert (len(extra) == len(cases)) if incremental else (not extra)
        statistics = {
            'cases': len(cases), 'all_ipc_frames': len(frames),
            'all_ipc_pipe_bytes': sum(row['request_frame_bytes'] + row['reply_frame_bytes']
                                      for row in frames),
            'medicine_request_pipe_bytes': sum(row['request_frame_bytes'] for row in selected),
            'medicine_reply_pipe_bytes': sum(row['reply_frame_bytes'] for row in selected),
            'medicine_reply_text_bytes': sum(row['reply_bytes'] for row in child_encoding),
            'ipc_round_trip_including_lookup': _timings(
                [row['round_trip_ns'] for row in selected]),
            'child_reply_json_encode': _timings(
                [row['json_encode_ns'] for row in child_encoding]),
            'parent_trace_reply_serialize': _timings(
                [row.serialization_ns for row in reply_costs]),
            'parent_context_parse_and_restore': _timings(
                [row['context_parse_ns'] + row['restore_ns'] for row in extra]) if extra else None,
            'exact_matches': sum(outcome['exact_match'] for outcome in outcomes),
            'overlap_cases': sum(bool(set(outcome['recommended_drugs']) &
                                      set(outcome['actual_drugs'])) for outcome in outcomes),
        }
        return statistics, outcomes, expert_inputs


def _worker(root: Path, cases: list, incremental: bool, sender):
    try:
        sender.send(('ok', _trial(root, cases, incremental)))
    except Exception as exc:
        sender.send(('error', f'{type(exc).__name__}: {exc}'))
    finally:
        sender.close()


def _isolated_trial(root: Path, cases: list, incremental: bool):
    """Run each mode with identical trace/message ID sequences."""
    context = mp.get_context('spawn')
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(root, cases, incremental, sender))
    process.start()
    sender.close()
    try:
        if not receiver.poll(90):
            process.terminate()
            raise TimeoutError('药物说明对照运行超时')
        status, result = receiver.recv()
    finally:
        receiver.close()
        process.join(timeout=5)
    if process.is_alive():
        process.terminate()
        process.join()
        raise RuntimeError('药物说明对照进程未退出')
    if status != 'ok' or process.exitcode != 0:
        raise RuntimeError(f'药物说明对照运行失败：{result}')
    return result


def measure() -> dict:
    cases = load_dialmed_cases(DIALMED_TRAIN_PATH, 10)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        full, full_outcomes, full_inputs = _isolated_trial(root / 'full', cases, False)
        delta, delta_outcomes, delta_inputs = _isolated_trial(root / 'delta', cases, True)
    saved = full['medicine_reply_pipe_bytes'] - delta['medicine_reply_pipe_bytes']
    total_saved = full['all_ipc_pipe_bytes'] - delta['all_ipc_pipe_bytes']
    return {
        'cases': len(cases), 'model': 'FixedModel; no gateway calls',
        'transport': 'multiprocessing Pipe; shared_dialog=False; incremental_history=False',
        'outputs_identical': full_outcomes == delta_outcomes,
        'expert_input_text_identical': full_inputs == delta_inputs,
        'full': full, 'incremental': delta,
        'medicine_reply_pipe_bytes_saved': saved,
        'medicine_reply_pipe_bytes_saved_percent': round(
            100 * saved / full['medicine_reply_pipe_bytes'], 2),
        'all_ipc_pipe_bytes_saved': total_saved,
        'all_ipc_pipe_bytes_saved_percent': round(
            100 * total_saved / full['all_ipc_pipe_bytes'], 2),
        'timing_notes': ('IPC RTT includes medicine lookup and serialization in the child, '
                         'but excludes main-process reconstruction after the reply. Child '
                         'JSON encoding, parent Trace encoding, and delta-only reconstruction '
                         'are reported separately; ten local samples are exploratory.'),
    }


if __name__ == '__main__':
    result = measure()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n',
                      encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
