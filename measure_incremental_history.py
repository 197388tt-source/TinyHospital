"""Compare full and incremental retrieval.history replies on ten fixed cases."""

from __future__ import annotations

import json
import multiprocessing as mp
import tempfile
from pathlib import Path
from statistics import mean, median

from cases import DIALMED_TRAIN_PATH, load_dialmed_cases
from hospital_process import ProcessHospitalSession
from performance_baseline import FixedModel


OUTPUT = Path(__file__).resolve().parent / 'runs/stage8_incremental_history.json'


def _paths(root: Path) -> dict:
    return {
        'golden_path': root / 'golden.json',
        'reflection_path': root / 'reflection.json',
        'communication_path': root / 'communication.jsonl',
        'control_path': root / 'control.jsonl',
        'llm_path': root / 'llm.jsonl',
        'dataset_path': DIALMED_TRAIN_PATH,
    }


def _statistics(values: list[int]) -> dict:
    return {'samples': len(values), 'total_us': round(sum(values) / 1000, 2),
            'mean_us': round(mean(values) / 1000, 2),
            'median_us': round(median(values) / 1000, 2)}


def _trial(root: Path, cases: list, incremental: bool) -> tuple[dict, list, list[str]]:
    with ProcessHospitalSession(
        retrieval_mode='jaccard', verbose=False, training=False, llm_client_factory=FixedModel,
        incremental_history=incremental, shared_dialog=False,
        **_paths(root),
    ) as session:
        outcomes = [session.run_case(case) for case in cases]
        frames = session.runtime.transport_frame_stats
        history_frames = [row for row in frames if row['capability'] == 'retrieval.history']
        history_reply_costs = [row for row in session.trace.packet_costs
            if row.capability == 'retrieval.history' and row.source == 'RetrievalAgent']
        doctor_texts = [packet.payload['text'] for packet in session.trace.packets
            if packet.payload.get('capability') == 'doctor.recommend'
            and packet.head.source == 'RetrievalAgent']
        history_texts = [packet.payload['text'] for packet in session.trace.packets
            if packet.payload.get('capability') == 'retrieval.history'
            and packet.head.source == 'RetrievalAgent']
        child_encoding = session.runtime.control_agent(
            'RetrievalAgent', 'snapshot')['history_response_encoding']
        extra = session.history_delta_stats
        assert len(history_frames) == len(history_reply_costs) == len(child_encoding) == len(cases)
        assert (len(extra) == len(cases)) if incremental else (not extra)
        return ({
            'case_count': len(cases), 'all_ipc_frames': len(frames),
            'all_ipc_pipe_bytes': sum(row['request_frame_bytes'] + row['reply_frame_bytes']
                                      for row in frames),
            'history_request_pipe_bytes': sum(row['request_frame_bytes']
                                              for row in history_frames),
            'history_reply_pipe_bytes': sum(row['reply_frame_bytes']
                                            for row in history_frames),
            'history_reply_text_bytes': sum(row['reply_bytes'] for row in child_encoding),
            'empty_history_replies': sum(not any(json.loads(text)['history'].values())
                                         for text in history_texts),
            'retrieval_ipc_round_trip': _statistics(
                [row['round_trip_ns'] for row in history_frames]),
            'child_reply_json_encode': _statistics(
                [row['json_encode_ns'] for row in child_encoding]),
            'parent_trace_reply_serialize': _statistics(
                [row.serialization_ns for row in history_reply_costs]),
            'parent_context_parse_and_restore': _statistics(
                [row['context_parse_ns'] + row['restore_ns'] for row in extra]) if extra else None,
        }, outcomes, doctor_texts)


def _trial_worker(root: Path, cases: list, incremental: bool, sender) -> None:
    try:
        sender.send(('ok', _trial(root, cases, incremental)))
    except Exception as exc:
        sender.send(('error', f'{type(exc).__name__}: {exc}'))
    finally:
        sender.close()


def _isolated_trial(root: Path, cases: list, incremental: bool):
    """Give both modes the same fresh trace and message ID sequence."""
    context = mp.get_context('spawn')
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_trial_worker,
                              args=(root, cases, incremental, sender))
    process.start()
    sender.close()
    try:
        if not receiver.poll(90):
            process.terminate()
            raise TimeoutError('对照运行超时')
        status, result = receiver.recv()
    finally:
        receiver.close()
        process.join(timeout=5)
    if process.is_alive():
        process.terminate()
        process.join()
        raise RuntimeError('对照进程未正常退出')
    if status != 'ok' or process.exitcode != 0:
        raise RuntimeError(f'对照运行失败：{result}')
    return result


def measure() -> dict:
    cases = load_dialmed_cases(DIALMED_TRAIN_PATH, 10)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        original, old_outputs, old_doctor = _isolated_trial(root / 'full', cases, False)
        incremental, new_outputs, new_doctor = _isolated_trial(root / 'incremental', cases, True)
    saved = original['history_reply_pipe_bytes'] - incremental['history_reply_pipe_bytes']
    return {
        'cases': len(cases), 'model': 'FixedModel; no gateway calls',
        'transport': 'multiprocessing Pipe; shared_dialog=False in both modes',
        'outputs_identical': old_outputs == new_outputs,
        'doctor_input_text_identical': old_doctor == new_doctor,
        'full': original, 'incremental': incremental,
        'history_reply_pipe_bytes_saved': saved,
        'history_reply_pipe_bytes_saved_percent': round(
            100 * saved / original['history_reply_pipe_bytes'], 2),
        'all_ipc_pipe_bytes_saved': (original['all_ipc_pipe_bytes']
                                     - incremental['all_ipc_pipe_bytes']),
        'all_ipc_pipe_bytes_saved_percent': round(
            100 * (original['all_ipc_pipe_bytes'] - incremental['all_ipc_pipe_bytes'])
            / original['all_ipc_pipe_bytes'], 2),
        'timing_notes': ('RTT includes history lookup and IPC; child JSON encode is the outgoing '
                         'reply only. Parent Trace serialization measures its additional reply '
                         'encoding; delta-only parent parse and reconstruction are reported '
                         'separately. Timings are local exploratory samples, not an isolated '
                         'serialization or final performance baseline.'),
    }


if __name__ == '__main__':
    result = measure()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n',
                      encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
