"""Retrieval agent for historical cases and medicine introductions."""

from __future__ import annotations

import json
from time import perf_counter_ns

from tool.medicine_store import MedicineKnowledgeBase
from history.memory import CaseMemoryStore
from tool.tracing import Trace

from .base import Agent, State


class RetrievalAgent(Agent):
    def __init__(self, trace: Trace, memories: CaseMemoryStore,
                 medicines: MedicineKnowledgeBase, incremental_history: bool = False,
                 incremental_medicines: bool = False, vector_transport: str = 'text') -> None:
        super().__init__('RetrievalAgent', trace)
        self.memories, self.medicines = memories, medicines
        self.vector_transport = vector_transport
        if type(incremental_history) is not bool or type(incremental_medicines) is not bool:
            raise ValueError('增量回复开关必须为布尔值')
        self.incremental_history = incremental_history
        self.incremental_medicines = incremental_medicines
        self.history_query_count = 0
        self.vector_transport_stats = {'received_vectors': 0, 'shared_read_bytes': 0, 'binary_bytes': 0}
        self.history_response_encoding: list[dict] = []
        self.medicine_response_encoding: list[dict] = []

    def retrieve_history(self, message: str, binary: bytes | None = None) -> str:
        self.receive_task(State.IDLE)
        data = json.loads(message)
        vector_only = self.vector_transport != 'text'
        expected = ({'case_id', 'dataset_path', 'count', 'query_embedding'} if vector_only else
                    {'case_id', 'dataset_path', 'patient_dialog', 'count'})
        if (not isinstance(data, dict) or set(data) != expected
                or type(data.get('case_id')) is not int or not isinstance(data.get('dataset_path'), str)
                or (not vector_only and (not isinstance(data.get('patient_dialog'), str)
                                         or not data['patient_dialog'].strip()))
                or type(data.get('count')) is not int or not 0 < data['count'] <= len(self.medicines.names)
                or 'actual_drugs' in data or 'drug_labels' in data):
            raise ValueError('历史检索请求字段无效')
        self.history_query_count += 1
        query_vector = None
        descriptor = data.get('query_embedding')
        if vector_only and (
                not isinstance(descriptor, dict) or descriptor.get('storage') != self.vector_transport):
            raise ValueError('查询向量缺失或传输模式不匹配，禁止静默重新编码')
        if descriptor is not None:
            from embedding.transport import receive_vector
            if self.memories.semantic is None or self.runtime is None:
                raise ValueError('查询向量需要 embedding 检索和有效 Runtime')
            if self.runtime.trace.packets[-1].head.source != 'PatientAgent':
                raise ValueError('查询向量必须由 PatientAgent 提交')
            query_vector = receive_vector(
                descriptor, trace_id=self.runtime.trace_id,
                fingerprint=self.memories.semantic.encoder.config.fingerprint,
                dialog=data.get('patient_dialog'), binary=binary)
            self.vector_transport_stats['received_vectors'] += 1
            key = 'shared_read_bytes' if descriptor['storage'] == 'shared_memory' else 'binary_bytes'
            self.vector_transport_stats[key] += descriptor['size']
        elif binary is not None:
            raise ValueError('二进制向量缺少协议描述')
        retrieved = self.memories.retrieve(
            data.get('patient_dialog'), data['case_id'], data['dataset_path'],
            query_vector=query_vector,
            query_text_hash=descriptor.get("text_hash") if descriptor is not None else None,
        )
        # Keep source dialogue, case IDs, summaries and scores inside Retrieval.
        history = {
            'golden_experiences': [row['experience'] for row in retrieved['golden_experiences']],
            'reflections': [row['reflection'] for row in retrieved['reflections']],
        }
        # A vector-only request cannot echo a dialogue that it never received.
        reply = ({'case_id': data['case_id'], 'history': history}
                 if self.incremental_history or vector_only else {**data, 'history': history})
        started = perf_counter_ns()
        encoded = json.dumps(reply, ensure_ascii=False)
        encoding_ns = perf_counter_ns() - started
        self.history_response_encoding.append({
            'case_id': data['case_id'], 'incremental': self.incremental_history,
            'reply_bytes': len(encoded.encode('utf-8')), 'json_encode_ns': encoding_ns,
        })
        self.transition(State.IDLE)
        return encoded

    def retrieve_medicines(self, message: str) -> str:
        self.receive_task(State.IDLE)
        assert self.runtime is not None
        if self.runtime.trace.packets[-1].head.source != 'ExpertAgent':
            raise ValueError('仅 ExpertAgent 可请求评判后的药物说明')
        data = json.loads(message)
        recommended, actual = data.get('recommended_drugs'), data.get('actual_drugs')
        if (not isinstance(recommended, list) or not isinstance(actual, list)
                or not recommended or not actual or any(not isinstance(name, str) or not name.strip()
                                                   for name in recommended + actual)):
            raise ValueError('药物检索请求无效')
        introductions = self.medicines.introductions_for(
            list(dict.fromkeys(recommended + actual)))
        response = ({'case_id': data['case_id'],
                     'medicine_introductions': introductions}
                    if self.incremental_medicines else {
                        'case_id': data['case_id'],
                        'recommended_drugs': recommended,
                        'actual_drugs': actual,
                        'medicine_introductions': introductions,
                    })
        started = perf_counter_ns()
        encoded = json.dumps(response, ensure_ascii=False)
        encoding_ns = perf_counter_ns() - started
        self.medicine_response_encoding.append({
            'case_id': data['case_id'], 'incremental': self.incremental_medicines,
            'reply_bytes': len(encoded.encode('utf-8')), 'json_encode_ns': encoding_ns,
        })
        self.transition(State.IDLE)
        return encoded
