"""Parent-orchestrated Tiny Hospital with one process per medical Agent."""

from __future__ import annotations

import json
import math
import secrets
from enum import Enum
from functools import partial
from pathlib import Path
from time import perf_counter_ns
from types import SimpleNamespace
from typing import Any, Callable

from agent import DoctorAgent, ExpertAgent, PatientAgent, RetrievalAgent, State
from cases import (DIALMED_TRAIN_PATH, DialMedCase, default_dataset_path,
                   memory_source_dataset_path)
from decision_vector import DecisionVectorCodec
from history.memory import CaseMemoryStore, GOLDEN_EXPERIENCE_PATH, REFLECTION_MEMORY_PATH
from ipc_runtime import BinaryReply, IpcRuntimeState, MultiprocessRuntime
from protocol import AgentPacket, MessageType
from shared_transport import SharedPayloadArena, read_reference, validate_reference, attach_shared_memory
from tool.llm_client import LLMClient
from tool.llm_usage import append_llm_usage
from tool.medicine_store import MedicineKnowledgeBase
from tool.tracing import Trace, append_communication_records


CAPABILITIES = {
    "PatientAgent": {
        "patient.provide_case_truth": "provide_case_truth",
        "patient.request_history": "request_history",
        "patient.provide_recommendation_context": "provide_recommendation_context",
    },
    "RetrievalAgent": {
        "retrieval.history": "retrieve_history",
        "retrieval.medicines": "retrieve_medicines",
    },
    "DoctorAgent": {"doctor.recommend": "recommend"},
    "ExpertAgent": {
        "expert.receive_case_truth": "receive_case_truth",
        "expert.evaluate": "evaluate",
        "expert.review": "review",
    },
}
FLOW_STEPS = {
    "patient.provide_case_truth": 0,
    "expert.receive_case_truth": 1,
    "patient.request_history": 2,
    "patient.provide_recommendation_context": 2,
    "retrieval.history": 3,
    "doctor.recommend": 4,
    "expert.evaluate": 5,
    "retrieval.medicines": 6,
    "expert.review": 7,
}

HISTORY_CONTEXT_FIELDS = {'case_id', 'dataset_path', 'patient_dialog', 'count'}
HISTORY_FIELDS = {'golden_experiences', 'reflections'}


def _history_context_from_patient(text: str, case: DialMedCase,
                                  dataset_path: str) -> dict[str, Any]:
    """Keep only the Patient fields that the original Doctor message contained."""
    context = json.loads(text)
    context.pop('query_embedding', None)
    if (not isinstance(context, dict) or set(context) != HISTORY_CONTEXT_FIELDS
            or type(context['case_id']) is not int
            or context['case_id'] != case.case_id
            or context['dataset_path'] != dataset_path
            or context['patient_dialog'] != case.patient_input
            or type(context['count']) is not int or context['count'] != case.drug_count):
        raise ValueError('Patient 病例上下文无效或包含额外字段')
    return context


def _history_context_from_case(case: DialMedCase, dataset_path: str) -> dict[str, Any]:
    """The parent alone retains dialogue while Retrieval receives a vector."""
    return {'case_id': case.case_id, 'dataset_path': dataset_path,
            'patient_dialog': case.patient_input, 'count': case.drug_count}


def _restore_history_response(context: dict[str, Any], text: str) -> str:
    """Recreate the exact original Doctor input from a checked delta reply."""
    delta = json.loads(text)
    if (not isinstance(delta, dict) or set(delta) != {'case_id', 'history'}
            or type(delta['case_id']) is not int
            or delta['case_id'] != context['case_id']
            or not isinstance(delta['history'], dict)
            or set(delta['history']) != HISTORY_FIELDS
            or any(not isinstance(items, list)
                   or any(not isinstance(item, str) or not item.strip() for item in items)
                   for items in delta['history'].values())):
        raise ValueError('Retrieval 增量回复无效或病例编号不匹配')
    return json.dumps({**context, 'history': delta['history']}, ensure_ascii=False)


def _medicine_context_from_expert(text: str, case: DialMedCase, trace_id: int,
                                  evaluation_message_id: int,
                                  binary: bytes, codec: DecisionVectorCodec) -> dict[str, Any]:
    """Save only the two drug lists already authorized by Expert's evaluation."""
    data = json.loads(text)
    recommended = data.get('recommended_drugs') if isinstance(data, dict) else None
    actual = data.get('actual_drugs') if isinstance(data, dict) else None
    if (not isinstance(data, dict)
            or set(data) != {'case_id', 'recommended_drugs', 'actual_drugs'}
            or type(data['case_id']) is not int or data['case_id'] != case.case_id
            or not isinstance(recommended, list) or not recommended
            or any(not isinstance(name, str) or not name.strip() for name in recommended)
            or not isinstance(actual, list) or actual != list(case.drug_labels)
            or recommended != codec.decode(binary)):
        raise ValueError('Expert 药物评判上下文无效或病例编号不匹配')
    return {'trace_id': trace_id, 'evaluation_message_id': evaluation_message_id,
            'case_id': data['case_id'], 'recommended_drugs': recommended,
            'actual_drugs': actual}


def _restore_medicine_response(context: dict[str, Any], text: str,
                               request: AgentPacket, reply: AgentPacket,
                               *, incremental: bool = True,
                               shared_medicines: bool = False) -> str:
    """Validate the reply's identity and restore Expert's original input."""
    if (request.head.trace_id != context['trace_id']
            or request.head.source != 'ExpertAgent'
            or request.head.correlation_id != context['evaluation_message_id']
            or request.payload.get('capability') != 'retrieval.medicines'
            or reply.head.trace_id != context['trace_id']
            or reply.head.correlation_id != request.head.message_id
            or reply.head.source != 'RetrievalAgent'
            or reply.head.message_type != MessageType.RESPONSE.value
            or reply.payload.get('capability') != 'retrieval.medicines'
            or ((reply.payload.get('data') != json.loads(text)) if 'data' in reply.payload
                else reply.payload.get('text') != text)):
        raise ValueError('Retrieval 药物说明回复的消息关联无效')
    delta = json.loads(text)
    introductions = delta.get('medicine_introductions') if isinstance(delta, dict) else None
    names = set(context['recommended_drugs']) | set(context['actual_drugs'])
    expected = ({'case_id', 'medicine_introductions'} if incremental else
                {'case_id', 'recommended_drugs', 'actual_drugs', 'medicine_introductions'})
    if shared_medicines:
        expected.add('shared_medicine_ref')
    if (not isinstance(delta, dict)
            or set(delta) != expected
            or type(delta['case_id']) is not int
            or delta['case_id'] != context['case_id']
            or (not incremental and (delta['recommended_drugs'] != context['recommended_drugs']
                                     or delta['actual_drugs'] != context['actual_drugs']))):
        raise ValueError('Retrieval 增量药物说明无效或病例编号不匹配')
    if shared_medicines:
        if introductions is not None:
            raise ValueError('Retrieval 共享药物说明占位符无效')
        validate_reference(delta['shared_medicine_ref'], context['trace_id'])
    elif (not isinstance(introductions, dict) or set(introductions) != names
          or any(not isinstance(item, dict) for item in introductions.values())):
        raise ValueError('Retrieval 增量药物说明无效或病例编号不匹配')
    restored = {
        'case_id': context['case_id'],
        'recommended_drugs': context['recommended_drugs'],
        'actual_drugs': context['actual_drugs'],
        'medicine_introductions': introductions,
    }
    if shared_medicines:
        restored['shared_medicine_ref'] = delta['shared_medicine_ref']
    return json.dumps(restored, ensure_ascii=False)


class AgentCasePhase(str, Enum):
    IDLE = "IDLE"
    CASE_READY = "CASE_READY"
    EXECUTING = "EXECUTING"
    FAILED = "FAILED"


class HospitalSessionState(str, Enum):
    STARTING = "STARTING"
    READY = "READY"
    RUNNING = "RUNNING"
    ABORTING = "ABORTING"
    FAILED = "FAILED"
    CLOSED = "CLOSED"


class _AgentTraceView:
    """Only context read by existing Agent methods; parent owns the full Trace."""

    def __init__(self) -> None:
        self.packets: list[AgentPacket] = []
        self.message_costs: list[Any] = []

    def state(self, _agent: str, _before: str, _after: str) -> None:
        pass


class HospitalAgentEndpoint:
    """Construct the original business Agent inside its child process."""

    def __init__(self, role: str, golden_path: Path, reflection_path: Path,
                 training: bool, memory_dataset: Path, dataset_path: str,
                 llm_client_factory: Callable[[], LLMClient],
                 incremental_history: bool = False,
                 incremental_medicines: bool = False,
                 binary_state: bool = True,
                 shared_medicines: bool = False,
                 structured_data: bool = False, retrieval_mode: str = 'embedding',
                 embedding_options: dict | None = None, vector_transport: str = 'shared_memory',
                 chat_timeout: float = 180.0) -> None:
        self.role = role
        self.training = training
        if retrieval_mode not in ('embedding', 'jaccard', 'none'):
            raise ValueError('retrieval_mode 必须为 embedding、jaccard 或 none')
        if training:
            retrieval_mode = 'none'
        if vector_transport not in ('text', 'binary', 'shared_memory'):
            raise ValueError('vector_transport 必须为 text、binary 或 shared_memory')
        self.vector_transport = vector_transport if retrieval_mode == 'embedding' else 'text'
        self.binary_state = binary_state
        self.shared_medicines = shared_medicines
        self.structured_data = structured_data
        self._medicine_arena = (SharedPayloadArena()
                                if training and shared_medicines and role == 'RetrievalAgent' else None)
        self._medicine_name: str | None = None
        self.trace = _AgentTraceView()
        medicines = MedicineKnowledgeBase()
        if role == "PatientAgent":
            self.agent = PatientAgent(self.trace, self.vector_transport, embedding_options)
        elif role == "RetrievalAgent":
            memories = CaseMemoryStore(golden_path, reflection_path, training, memory_dataset,
                                       retrieval_mode, embedding_options)
            self.agent = RetrievalAgent(self.trace, memories, medicines,
                                        incremental_history=incremental_history,
                                        incremental_medicines=incremental_medicines,
                                        vector_transport=self.vector_transport)
        elif role == "DoctorAgent":
            self.agent = DoctorAgent(self.trace, medicines, llm_client_factory, training=training)
        elif role == "ExpertAgent":
            memories = CaseMemoryStore(golden_path, reflection_path, training, memory_dataset,
                                       retrieval_mode, embedding_options)
            self.agent = ExpertAgent(self.trace, memories, medicines, dataset_path,
                                     llm_client_factory)
        else:
            raise ValueError(f"未知 Agent：{role}")
        if role in ('DoctorAgent', 'ExpertAgent'):
            if hasattr(self.agent.llm, 'timeout'):
                self.agent.llm.timeout = chat_timeout
            self.agent.diagnostics_path = (embedding_options or {}).get('diagnostics_path')
            self.agent.diagnostics_verbose = (embedding_options or {}).get('diagnostics_verbose', False)
        self.agent.runtime = SimpleNamespace(trace=self.trace, trace_id=None)
        self.active_trace: int | None = None
        self.phase = AgentCasePhase.IDLE
        self.decision_codec = (DecisionVectorCodec() if binary_state and role in
                               ("DoctorAgent", "ExpertAgent") else None)
        self.codec_events: list[dict[str, Any]] = []
        self.transport_codec_stats = {"request_decode_ns": 0, "response_encode_ns": 0}

    def handle(self, packet: AgentPacket, binary: bytes | None = None) -> str | BinaryReply:
        capability = packet.payload["capability"]
        if self.phase != AgentCasePhase.CASE_READY or packet.head.trace_id != self.active_trace:
            raise RuntimeError("Agent 病例 trace_id 或状态不匹配")
        method = CAPABILITIES[self.role][capability]
        self.trace.packets[:] = [packet]
        self.trace.message_costs[:] = [SimpleNamespace(step=FLOW_STEPS[capability])]
        self.phase = AgentCasePhase.EXECUTING
        try:
            text = (json.dumps(packet.payload["data"], ensure_ascii=False)
                    if "data" in packet.payload else packet.payload["text"])
            if capability == "expert.evaluate" and self.binary_state:
                if binary is None:
                    raise ValueError("Expert 推荐结果缺少二进制决策状态")
                started = perf_counter_ns()
                assert self.decision_codec is not None
                names = self.decision_codec.decode(binary)
                decode_ns = perf_counter_ns() - started
                recommendation = json.loads(text)
                expected = {"case_id", "reasoning_summary"} if self.training else {"case_id"}
                if not isinstance(recommendation, dict) or set(recommendation) != expected:
                    raise ValueError("Expert 推荐结果文本字段无效")
                recommendation["recommended_drugs"] = names
                text = json.dumps(recommendation, ensure_ascii=False)
                self.codec_events.append({"operation": "decode", "trace_id": packet.head.trace_id,
                                          "case_id": recommendation["case_id"],
                                          "binary_bytes": len(binary), "codec_ns": decode_ns})
            elif binary is not None and capability != "retrieval.history":
                raise ValueError("该 Agent capability 不接受二进制载荷")
            if capability == 'expert.review' and self.shared_medicines:
                details = json.loads(text)
                if (not isinstance(details, dict)
                        or set(details) != {'case_id', 'recommended_drugs', 'actual_drugs',
                                            'medicine_introductions', 'shared_medicine_ref'}
                        or details['medicine_introductions'] is not None
                        or type(details['case_id']) is not int):
                    raise ValueError('Expert 共享药物说明请求格式无效')
                data, _read_ns = read_reference(details.pop('shared_medicine_ref'),
                                                packet.head.trace_id)
                details['medicine_introductions'] = json.loads(data)
                text = json.dumps(details, ensure_ascii=False)
            if capability == 'retrieval.history':
                body = json.loads(text)
                if self.vector_transport != 'text':
                    descriptor = body.get('query_embedding')
                    if not isinstance(descriptor, dict) or descriptor.get('storage') != self.vector_transport:
                        raise ValueError('Patient 查询向量缺失或传输模式不匹配')
                result = self.agent.retrieve_history(text, binary=binary)
            else:
                result = getattr(self.agent, method)(text)
            if capability == 'patient.request_history' and self.vector_transport == 'binary':
                result = BinaryReply(result, self.agent.query_binary)
            if capability == 'retrieval.medicines' and self.shared_medicines:
                details = json.loads(result)
                if (not isinstance(details, dict) or type(details.get('case_id')) is not int
                        or not isinstance(details.get('medicine_introductions'), dict)):
                    raise ValueError('Retrieval 药物说明回复格式无效')
                assert self._medicine_arena is not None
                if self._medicine_name is None:
                    raise RuntimeError('Retrieval 药物共享段尚未预留')
                data = json.dumps(details['medicine_introductions'], ensure_ascii=False).encode('utf-8')
                ref, _write_ns, _written = self._medicine_arena.publish(
                    data, packet.head.trace_id, name=self._medicine_name)
                details['medicine_introductions'] = None
                details['shared_medicine_ref'] = ref
                result = json.dumps(details, ensure_ascii=False)
            if capability == "doctor.recommend" and self.binary_state:
                recommendation = json.loads(result)
                expected = {"case_id", "recommended_drugs"}
                if self.training:
                    expected.add("reasoning_summary")
                if not isinstance(recommendation, dict) or set(recommendation) != expected:
                    raise ValueError("Doctor 推荐结果字段无效")
                started = perf_counter_ns()
                assert self.decision_codec is not None
                vector = self.decision_codec.encode(recommendation["recommended_drugs"])
                encode_ns = perf_counter_ns() - started
                compact_text = json.dumps({key: value for key, value in recommendation.items()
                                           if key != "recommended_drugs"}, ensure_ascii=False)
                self.codec_events.append({"operation": "encode", "trace_id": packet.head.trace_id,
                                          "case_id": recommendation["case_id"],
                                          "original_json_bytes": len(result.encode("utf-8")),
                                          "compact_text_bytes": len(compact_text.encode("utf-8")),
                                          "binary_bytes": len(vector), "codec_ns": encode_ns})
                result = BinaryReply(compact_text, vector)
        except Exception:
            self.phase = AgentCasePhase.FAILED
            raise
        self.phase = AgentCasePhase.CASE_READY
        return result

    def control(self, op: str, payload: Any) -> Any:
        if op == "begin_case":
            if (self.phase != AgentCasePhase.IDLE or self.active_trace is not None
                    or self.agent.state != State.IDLE):
                raise RuntimeError("Agent 尚未结束上一病例")
            trace_id = payload["trace_id"]
            if type(trace_id) is not int or trace_id <= 0:
                raise ValueError("病例 trace_id 无效")
            if self.role == "PatientAgent":
                case = payload["case"]
                self.agent.vector_name = payload.get("query_vector_name")
                self.agent.begin_case(
                    DialMedCase(case["case_id"], case["patient_input"], tuple(case["drug_labels"])),
                    payload["dataset_path"],
                )
            if self._medicine_arena is not None:
                name = payload.get('shared_medicine_name')
                if (not isinstance(name, str) or len(name) != 38
                        or not name.startswith('thipc_')
                        or any(ch not in '0123456789abcdef' for ch in name[6:])):
                    raise ValueError('Retrieval 药物共享段名无效')
                self._medicine_name = name
            self.active_trace = trace_id
            self.agent.runtime.trace_id = trace_id
            self.phase = AgentCasePhase.CASE_READY
            return True
        if op == "snapshot":
            result = {"state": self.agent.state.value, "phase": self.phase.value,
                      "trace_id": self.active_trace,
                      "transport_codec_stats": dict(self.transport_codec_stats)}
            if self.role in ("DoctorAgent", "ExpertAgent"):
                result["model_calls"] = list(self.agent.model_calls)
                result["prompts"] = list(getattr(self.agent.llm, "prompts", []))
                result["codec_events"] = list(self.codec_events)
            if self.role == "PatientAgent" and self.agent.vector_sender is not None:
                result["query_vector_statistics"] = dict(self.agent.vector_sender.stats)
            if self.role == "RetrievalAgent":
                result["vector_transport_statistics"] = dict(self.agent.vector_transport_stats)
                result["retrieval_statistics"] = self.agent.memories.retrieval_statistics
                result["history_response_encoding"] = list(self.agent.history_response_encoding)
                result["medicine_response_encoding"] = list(self.agent.medicine_response_encoding)
            if self.role == "ExpertAgent":
                result['review_events'] = list(self.agent.review_events)
            return result
        if op in ("finish_case", "abort_case"):
            if op == "finish_case" and (self.phase != AgentCasePhase.CASE_READY
                                        or self.agent.state != State.IDLE):
                raise RuntimeError("Agent 病例结束时状态不正确")
            if op == "abort_case" and self.agent.state != State.IDLE:
                self.agent.state = State.IDLE
            if self.role in ("PatientAgent", "DoctorAgent", "ExpertAgent"):
                self.agent.clear_case()
            if self._medicine_arena is not None:
                self._medicine_arena.release()
                self._medicine_name = None
            self.active_trace = None
            self.agent.runtime.trace_id = None
            self.phase = AgentCasePhase.IDLE
            self.trace.packets.clear()
            self.trace.message_costs.clear()
            self.codec_events.clear()
            return True
        raise ValueError(f"未知 Agent 管理操作：{op}")

    def close(self) -> None:
        if self.role == "PatientAgent":
            self.agent.clear_case()
        if self._medicine_arena is not None:
            self._medicine_arena.release()
            self._medicine_name = None


class ProcessHospitalSession:
    """Run stage-specific routes with four persistent Agent processes."""

    def __init__(self, verbose: bool = True, golden_path: Path | None = None,
                 communication_path: Path | None = None, control_path: Path | None = None,
                 llm_path: Path | None = None, reflection_path: Path | None = None,
                 dataset_path: Path | None = None, training: bool = True,
                 llm_client_factory: Callable[[], LLMClient] = LLMClient,
                 timeout: float = 720.0, startup_timeout: float = 30.0,
                 shared_dialog: bool = False, shared_min_bytes: int = 256,
                 incremental_history: bool = False,
                 incremental_medicines: bool = False,
                 binary_state: bool = True,
                 shared_medicines: bool = False,
                 structured_data: bool = False, retrieval_mode: str = 'embedding',
                 embedding_options: dict | None = None, vector_transport: str = 'shared_memory',
                 chat_timeout: float = 180.0) -> None:
        dataset_path = Path(dataset_path) if dataset_path is not None else default_dataset_path(training)
        if retrieval_mode not in ('embedding', 'jaccard', 'none'):
            raise ValueError('retrieval_mode 必须为 embedding、jaccard 或 none')
        if training:
            retrieval_mode = 'none'
        if not math.isfinite(chat_timeout) or chat_timeout <= 0:
            raise ValueError('chat_timeout 必须为有限正数')
        if (type(shared_dialog) is not bool or type(shared_min_bytes) is not int
                or not 0 < shared_min_bytes <= 64 * 1024 * 1024
                or type(incremental_history) is not bool
                or type(incremental_medicines) is not bool
                or type(binary_state) is not bool
                or type(shared_medicines) is not bool
                or type(structured_data) is not bool):
            raise ValueError("共享对话、增量回复开关或阈值无效")
        if not binary_state and (incremental_medicines or shared_medicines):
            raise ValueError("普通 JSON 推荐消融暂不与药物增量回复组合")
        if vector_transport not in ('text', 'binary', 'shared_memory'):
            raise ValueError('vector_transport 必须为 text、binary 或 shared_memory')
        self.vector_transport = vector_transport if retrieval_mode == 'embedding' else 'text'
        self._query_vector_name = None
        root = Path(__file__).resolve().parent
        self.trace = Trace(verbose=verbose)
        self.training = training
        self.shared_dialog = shared_dialog
        self.shared_min_bytes = shared_min_bytes
        self.incremental_history = incremental_history
        self.incremental_medicines = incremental_medicines
        self.shared_medicines = shared_medicines
        self.binary_state = binary_state
        self.structured_data = structured_data
        self._history_context: dict[str, Any] | None = None
        self._medicine_context: dict[str, Any] | None = None
        self._medicine_codec = DecisionVectorCodec() if incremental_medicines or shared_medicines else None
        self.dataset_path = str(dataset_path.resolve())
        self.communication_path = communication_path or root / "runs/token_usage.jsonl"
        self.control_path = control_path or root / "runs/control_communication.jsonl"
        self.llm_path = llm_path or root / "runs/llm_usage.jsonl"
        embedding_options = dict(embedding_options or {})
        embedding_options.setdefault('diagnostics_path', self.llm_path.parent / 'diagnostics.jsonl')
        embedding_options.setdefault('diagnostics_verbose', verbose)
        self.golden_path = golden_path or GOLDEN_EXPERIENCE_PATH
        self.reflection_path = reflection_path or REFLECTION_MEMORY_PATH
        memory_dataset = memory_source_dataset_path(dataset_path, training)
        self.runtime = MultiprocessRuntime(
            self.trace, timeout=timeout, startup_timeout=startup_timeout,
            binary_required=binary_state, shared_medicines=shared_medicines)
        self._closed = False
        self.state = HospitalSessionState.STARTING
        self.last_error: str | None = None
        self.last_cleanup_error: str | None = None
        self.failed_case_ids: set[int] = set()
        self.decision_codec_stats: list[dict[str, Any]] = []
        self.history_delta_stats: list[dict[str, int]] = []
        self.medicine_delta_stats: list[dict[str, int]] = []
        try:
            for role, handlers in CAPABILITIES.items():
                factory = partial(HospitalAgentEndpoint, role, self.golden_path,
                                  self.reflection_path, training, memory_dataset,
                                  self.dataset_path, llm_client_factory,
                                  incremental_history, incremental_medicines,
                                  binary_state, shared_medicines, structured_data,
                                  retrieval_mode, embedding_options, self.vector_transport, chat_timeout)
                self.runtime.register_managed_agent(role, tuple(handlers), factory)
            append_communication_records(self.control_path, self.trace.packet_costs)
            self.state = HospitalSessionState.READY
        except Exception:
            self.runtime.close()
            self.state = HospitalSessionState.CLOSED
            raise

    def _deliver(self, kind: MessageType, source: str, capability: str,
                 text: str, correlation_id: int | None = None,
                 binary: bytes | None = None,
                 shared_fields: tuple[str, ...] | None = None) -> tuple[AgentPacket, Any]:
        structured = None
        if self.structured_data:
            try:
                candidate = json.loads(text)
            except (TypeError, ValueError):
                candidate = None
            if isinstance(candidate, dict):
                structured = candidate
        packet = AgentPacket.create(
            trace_id=self.runtime.trace_id, message_type=kind, source=source,
            correlation_id=correlation_id,
            payload={"capability": capability,
                     "data": structured} if structured is not None else
                    {"capability": capability, "text": text},
        )
        if binary is not None and shared_fields is not None:
            raise ValueError("同一投递暂不同时使用二进制侧载荷与共享字段")
        if binary is not None:
            delivery = self.runtime.deliver_binary(packet, binary)
        elif shared_fields is not None:
            delivery = self.runtime.deliver_shared(
                packet, fields=shared_fields, min_bytes=self.shared_min_bytes)
        else:
            delivery = self.runtime.deliver(packet)
        if delivery.packet.head.message_type == MessageType.ERROR.value:
            raise RuntimeError(delivery.packet.payload["error"])
        if capability != "case.complete" and delivery.packet.head.message_type != MessageType.RESPONSE.value:
            raise RuntimeError("Agent 未返回确认报文")
        return packet, delivery.output

    def run_case(self, case: DialMedCase) -> dict:
        if not isinstance(case, DialMedCase):
            raise TypeError("必须提供 DialMedCase")
        if self.state != HospitalSessionState.READY or self.runtime.trace_id is not None:
            raise RuntimeError(f"会话当前状态为 {self.state.value}，无法开始病例")
        if self._history_context is not None or self._medicine_context is not None:
            raise RuntimeError('上一病例的临时上下文尚未清理')
        if case.case_id in self.failed_case_ids:
            raise RuntimeError("失败病例不可在同一会话重试；请先核对已写入的记忆")
        trace_id = self.runtime.start_case()
        self.trace.begin_case(trace_id)
        self.state = HospitalSessionState.RUNNING
        try:
            medicine_name = (self.runtime.reserve_medicine_segment()
                             if self.training and self.shared_medicines else None)
            if self.vector_transport == 'shared_memory':
                self._query_vector_name = 'thipc_' + secrets.token_hex(16)
            for role in CAPABILITIES:
                payload = {"trace_id": trace_id}
                if role == "PatientAgent":
                    payload["query_vector_name"] = self._query_vector_name
                    payload.update({"case": {
                        "case_id": case.case_id, "patient_input": case.patient_input,
                        "drug_labels": list(case.drug_labels),
                    }, "dataset_path": self.dataset_path})
                if role == 'RetrievalAgent' and medicine_name is not None:
                    payload['shared_medicine_name'] = medicine_name
                self.runtime.control_agent(role, "begin_case", payload)

            truth, truth_text = self._deliver(MessageType.REQUEST, "System",
                                              "patient.provide_case_truth", "请将病例标签私下交给专家。")
            self._deliver(MessageType.REQUEST, "PatientAgent", "expert.receive_case_truth",
                          truth_text, truth.head.message_id,
                          shared_fields=("/patient_dialog",) if self.shared_dialog else None)
            if self.training:
                request, request_text = self._deliver(
                    MessageType.REQUEST, "System", "patient.provide_recommendation_context",
                    "请提供对话和目标药物数量。")
                recommendation, recommendation_result = self._deliver(
                    MessageType.REQUEST, "PatientAgent", "doctor.recommend",
                    request_text, request.head.message_id,
                    shared_fields=("/patient_dialog",) if self.shared_dialog else None)
            else:
                request, request_text = self._deliver(MessageType.REQUEST, "System",
                                                      "patient.request_history", "请提供对话和目标药物数量。")
                query_binary = None
                if isinstance(request_text, BinaryReply):
                    query_binary, request_text = request_text.binary, request_text.text
                if self.vector_transport != 'text':
                    descriptor = json.loads(request_text).get('query_embedding')
                    from protocol import validate_embedding_descriptor
                    validate_embedding_descriptor(descriptor, trace_id)
                    if descriptor['storage'] != self.vector_transport:
                        raise ValueError('Patient 查询向量模式不匹配')
                    if self.vector_transport == 'shared_memory':
                        ref = descriptor['reference']
                        validate_reference(ref, trace_id)
                        if ref['name'] != self._query_vector_name:
                            raise ValueError('Patient 未使用父进程预留的向量共享段')
                    elif query_binary is None:
                        raise ValueError('Patient 未返回二进制查询向量')
                restore_vector_history = self.vector_transport != 'text'
                if self.incremental_history or restore_vector_history:
                    context_started = perf_counter_ns()
                    self._history_context = (_history_context_from_case(case, self.dataset_path)
                                             if restore_vector_history else
                                             _history_context_from_patient(
                                                 request_text, case, self.dataset_path))
                    context_parse_ns = perf_counter_ns() - context_started
                history, history_text = self._deliver(MessageType.REQUEST, "PatientAgent",
                                                      "retrieval.history", request_text, request.head.message_id,
                                                      binary=query_binary,
                                                      shared_fields=("/patient_dialog",) if self.shared_dialog and self.vector_transport == 'text' else None)
                if self.incremental_history or restore_vector_history:
                    if self._history_context is None:
                        raise RuntimeError('Retrieval 回复缺少已保存的病例上下文')
                    restore_started = perf_counter_ns()
                    history_text = _restore_history_response(self._history_context, history_text)
                    restore_ns = perf_counter_ns() - restore_started
                    if self.incremental_history:
                        self.history_delta_stats.append({
                            'case_id': case.case_id, 'context_parse_ns': context_parse_ns,
                            'restore_ns': restore_ns,
                        })
                    self._history_context = None
                recommendation, recommendation_result = self._deliver(
                    MessageType.RESPONSE, "RetrievalAgent", "doctor.recommend",
                    history_text, history.head.message_id,
                    shared_fields=("/patient_dialog",) if self.shared_dialog else None)
            if self.binary_state:
                if not isinstance(recommendation_result, BinaryReply):
                    raise RuntimeError("Doctor 未返回二进制决策状态")
                evaluation, evaluation_text = self._deliver(
                    MessageType.REQUEST, "DoctorAgent", "expert.evaluate",
                    recommendation_result.text, recommendation.head.message_id,
                    binary=recommendation_result.binary)
            else:
                if not isinstance(recommendation_result, str):
                    raise RuntimeError("Doctor 未返回完整 JSON 推荐")
                evaluation, evaluation_text = self._deliver(
                    MessageType.REQUEST, "DoctorAgent", "expert.evaluate",
                    recommendation_result, recommendation.head.message_id)
            completed, completed_text = evaluation, evaluation_text
            if self.training:
                if self.incremental_medicines or self.shared_medicines:
                    context_started = perf_counter_ns()
                    assert self._medicine_codec is not None
                    self._medicine_context = _medicine_context_from_expert(
                        evaluation_text, case, trace_id, evaluation.head.message_id,
                        recommendation_result.binary, self._medicine_codec)
                    context_parse_ns = perf_counter_ns() - context_started
                introductions, introductions_text = self._deliver(
                    MessageType.REQUEST, "ExpertAgent", "retrieval.medicines",
                    evaluation_text, evaluation.head.message_id)
                if self.incremental_medicines or self.shared_medicines:
                    if self._medicine_context is None:
                        raise RuntimeError('Retrieval 回复缺少已保存的药物评判上下文')
                    restore_started = perf_counter_ns()
                    introductions_text = _restore_medicine_response(
                        self._medicine_context, introductions_text,
                        introductions, self.trace.packets[-1],
                        incremental=self.incremental_medicines,
                        shared_medicines=self.shared_medicines)
                    restore_ns = perf_counter_ns() - restore_started
                    self.medicine_delta_stats.append({
                        'case_id': case.case_id, 'context_parse_ns': context_parse_ns,
                        'restore_ns': restore_ns,
                    })
                    self._medicine_context = None
                completed, completed_text = self._deliver(
                    MessageType.RESPONSE, "RetrievalAgent", "expert.review",
                    introductions_text, introductions.head.message_id)
            outcome = json.loads(completed_text)
            if outcome.get('reflection_error'):
                print(f"[反思未保存] 病例 {case.case_id}：{outcome['reflection_error']}；保留评判结果，继续运行。", flush=True)
            doctor = self.runtime.control_agent("DoctorAgent", "snapshot")
            expert = self.runtime.control_agent("ExpertAgent", "snapshot")
            append_llm_usage(self.llm_path, case.case_id,
                             doctor["model_calls"] + expert["model_calls"])
            completion_message = "病例评判与经验提炼已完成。" if self.training else "病例评判已完成。"
            self._deliver(MessageType.RESPONSE, "ExpertAgent", "case.complete",
                          completion_message, completed.head.message_id)
            for role in CAPABILITIES:
                snapshot = self.runtime.control_agent(role, "snapshot")
                if (snapshot["state"] != State.IDLE.value
                        or snapshot["phase"] != AgentCasePhase.CASE_READY.value):
                    raise RuntimeError(f"{role} 病例结束后未回到 IDLE")
            for role in CAPABILITIES:
                self.runtime.control_agent(role, "finish_case")
            self.decision_codec_stats.extend(doctor["codec_events"] + expert["codec_events"])
            assert self.trace._case_packet_start is not None
            case_packets = self.trace.packet_costs[self.trace._case_packet_start:]
            self.trace.finish_case(trace_id)
            append_communication_records(self.communication_path, case_packets, case.case_id)
            self.runtime.finish_case()
            self.state = HospitalSessionState.READY
            return outcome
        except Exception as exc:
            self.last_error = str(exc)
            self.failed_case_ids.add(case.case_id)
            self.state = HospitalSessionState.ABORTING
            if self.runtime.state == IpcRuntimeState.CASE_ACTIVE:
                for role in CAPABILITIES:
                    try:
                        self.runtime.control_agent(role, "abort_case")
                    except RuntimeError as cleanup_exc:
                        self.last_cleanup_error = str(cleanup_exc)
                        self.runtime.fail(f"病例中止失败：{cleanup_exc}")
                        break
            if self.trace._case_packet_start is not None:
                try:
                    self.trace.finish_case(trace_id)
                except Exception as cleanup_exc:
                    self.last_cleanup_error = str(cleanup_exc)
                    self.runtime.fail(f"病例统计清理失败：{cleanup_exc}")
            if self.runtime.trace_id is not None:
                try:
                    self.runtime.finish_case()
                except Exception as cleanup_exc:
                    self.last_cleanup_error = str(cleanup_exc)
                    self.runtime.fail(f"Runtime 病例清理失败：{cleanup_exc}")
            self.state = (HospitalSessionState.FAILED
                          if self.runtime.state == IpcRuntimeState.FAILED
                          else HospitalSessionState.READY)
            raise
        finally:
            self._release_query_vector()
            self._history_context = None
            self._medicine_context = None

    def _release_query_vector(self):
        # Name is reserved before the child allocates: cleanup works even if the
        # Patient dies before sending its descriptor back to the parent.
        name, self._query_vector_name = self._query_vector_name, None
        if name is None:
            return
        try:
            segment = attach_shared_memory(name)
        except FileNotFoundError:
            return
        try:
            segment.unlink()
        except FileNotFoundError:
            pass
        finally:
            segment.close()

    @property
    def retrieval_statistics(self) -> dict:
        snapshot = self.runtime.control_agent("RetrievalAgent", "snapshot")
        stats = snapshot["retrieval_statistics"]
        stats['vector_transport'] = self.vector_transport
        stats['vector_receive'] = snapshot['vector_transport_statistics']
        patient = self.runtime.control_agent("PatientAgent", "snapshot")
        stats['query_sender'] = patient.get('query_vector_statistics')
        return stats

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self.trace.verbose and self.state == HospitalSessionState.READY:
                statistics = self.retrieval_statistics
                self.trace.print_overall_summary()
                print("历史检索统计：" + json.dumps(statistics, ensure_ascii=False))
        finally:
            try:
                if self.trace._case_packet_start is not None and self.runtime.trace_id is not None:
                    self.trace.finish_case(self.runtime.trace_id)
            finally:
                try:
                    self.runtime.close()
                finally:
                    self._history_context = None
                    self._medicine_context = None
                    self._release_query_vector()
                    self._closed = True
                    self.state = HospitalSessionState.CLOSED

    def __enter__(self) -> ProcessHospitalSession:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()
