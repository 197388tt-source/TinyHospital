"""Exploratory byte, allocation and latency measurements for Stage 7."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from statistics import mean, median

from cases import DIALMED_TRAIN_PATH, load_dialmed_cases
from hospital_process import ProcessHospitalSession
from ipc_runtime import MultiprocessRuntime
from performance_baseline import FixedModel
from protocol import AgentPacket, MessageType
from tool.tracing import Trace


SELECTED = {'expert.receive_case_truth', 'retrieval.history', 'doctor.recommend'}


def _paths(root: Path) -> dict:
    return {
        'golden_path': root / 'golden.json',
        'reflection_path': root / 'reflections.json',
        'communication_path': root / 'communications.jsonl',
        'control_path': root / 'control.jsonl',
        'llm_path': root / 'llm.jsonl',
        'dataset_path': DIALMED_TRAIN_PATH,
    }


def _echo_length(value: str) -> str:
    return str(len(value))


def _echo_trial(shared: bool, count: int = 40) -> dict:
    trace = Trace(verbose=False)
    with MultiprocessRuntime(trace) as runtime:
        runtime.register_process_agent('EchoAgent', 'demo.echo', _echo_length)
        runtime.start_case()
        trace.begin_case(runtime.trace_id)
        value = json.dumps({'dialog': '重复对话' * 1400}, ensure_ascii=False)
        for _ in range(count):
            packet = AgentPacket.create(
                trace_id=runtime.trace_id, message_type=MessageType.REQUEST,
                source='System', payload={'capability': 'demo.echo', 'text': value},
            )
            delivery = (runtime.deliver_shared(packet, fields=('/dialog',), min_bytes=256)
                        if shared else runtime.deliver(packet))
            assert delivery.output == str(len(value))
        rows = runtime.transport_frame_stats
        runtime.finish_case()
        allocation = runtime.shared_memory_case_stats[-1]
    return {
        'samples': count,
        'median_round_trip_us': round(median(row['round_trip_ns'] for row in rows) / 1000, 2),
        'mean_round_trip_us': round(mean(row['round_trip_ns'] for row in rows) / 1000, 2),
        'pipe_bytes': sum(row['request_frame_bytes'] + row['reply_frame_bytes'] for row in rows),
        'shared_allocations': allocation['allocations'],
        'peak_shared_bytes': allocation['peak_bytes'],
        'shared_write_bytes': allocation['write_bytes'],
        'shared_read_bytes': sum(row['shared_read_bytes'] for row in rows),
        'shared_write_us': round(allocation['write_ns'] / 1000, 2),
        'shared_read_us': round(sum(row['shared_read_ns'] for row in rows) / 1000, 2),
    }


def measure() -> dict:
    cases = load_dialmed_cases(DIALMED_TRAIN_PATH, 10)
    modes = {}
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for label, enabled in (('pipe', False), ('shared_dialog', True)):
            with ProcessHospitalSession(
                retrieval_mode='jaccard', verbose=False, training=False, llm_client_factory=FixedModel,
                shared_dialog=enabled, **_paths(root / label),
            ) as session:
                outcomes = [session.run_case(case) for case in cases]
                all_rows = session.runtime.transport_frame_stats
                rows = [row for row in all_rows
                        if row['capability'] in SELECTED]
                allocations = session.runtime.shared_memory_case_stats
                modes[label] = {
                    'outcomes': outcomes,
                    'frames': len(rows),
                    'all_ipc_frames': len(all_rows),
                    'shared_frames': sum(row['mode'] == 'shared' for row in rows),
                    'pipe_bytes': sum(row['request_frame_bytes'] + row['reply_frame_bytes'] for row in rows),
                    'all_ipc_pipe_bytes': sum(row['request_frame_bytes'] + row['reply_frame_bytes']
                                              for row in all_rows),
                    'request_pipe_bytes': sum(row['request_frame_bytes'] for row in rows),
                    'mean_round_trip_ms_including_handler': round(
                        mean(row['round_trip_ns'] for row in rows) / 1e6, 3),
                    'shared_allocations': sum(row['allocations'] for row in allocations),
                    'peak_shared_bytes_max_case': max(row['peak_bytes'] for row in allocations),
                    'shared_write_bytes': sum(row['write_bytes'] for row in allocations),
                    'shared_read_bytes': sum(row['shared_read_bytes'] for row in rows),
                    'shared_write_us': round(sum(row['write_ns'] for row in allocations) / 1000, 2),
                    'shared_read_us': round(sum(row['shared_read_ns'] for row in rows) / 1000, 2),
                    'active_shared_bytes_after_cases': session.runtime.shared_memory_active_bytes,
                    'cleanup_errors': sum(len(row['cleanup_errors']) for row in allocations),
                }
    same = modes['pipe'].pop('outcomes') == modes['shared_dialog'].pop('outcomes')
    direct, shared = modes['pipe'], modes['shared_dialog']
    return {
        'cases': len(cases), 'outputs_identical': same,
        'hospital_handoff': modes,
        'hospital_pipe_bytes_saved': direct['pipe_bytes'] - shared['pipe_bytes'],
        'hospital_pipe_bytes_saved_percent': round(
            100 * (direct['pipe_bytes'] - shared['pipe_bytes']) / direct['pipe_bytes'], 2),
        'all_ipc_pipe_bytes_saved': direct['all_ipc_pipe_bytes'] - shared['all_ipc_pipe_bytes'],
        'all_ipc_pipe_bytes_saved_percent': round(
            100 * (direct['all_ipc_pipe_bytes'] - shared['all_ipc_pipe_bytes'])
            / direct['all_ipc_pipe_bytes'], 2),
        'no_op_echo_40_repeated_requests': {
            'pipe': _echo_trial(False), 'shared': _echo_trial(True),
        },
        'notes': ('Hospital RTT includes deterministic Agent work; echo RTT uses a trivial handler. '
                  'Copy timers cover shared-memory write/read only, not JSON encoding or Pipe copying. '
                  'This is an exploratory local measurement, not the final performance baseline.'),
    }


if __name__ == '__main__':
    result = measure()
    output = Path(__file__).resolve().parent / 'runs/stage7_shared_transport.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
