"""Patient agent: provide the case and privately deliver labels to Expert."""

from __future__ import annotations

import json

from cases import DialMedCase
from tool.tracing import Trace

from .base import Agent, State


class PatientAgent(Agent):
    def __init__(self, trace: Trace, vector_transport: str = 'text', embedding_options=None) -> None:
        super().__init__('PatientAgent', trace)
        self.case: DialMedCase | None = None
        self.dataset_path = ''
        self.vector_sender = None
        self.vector_name = None
        self.query_binary = None
        if vector_transport != 'text':
            from embedding.transport import QueryVectorSender
            self.vector_sender = QueryVectorSender(vector_transport, embedding_options)

    def begin_case(self, case: DialMedCase, dataset_path: str) -> None:
        if self.state != State.IDLE or self.case is not None:
            raise RuntimeError('PatientAgent 尚未结束上一病例')
        if not case.patient_input.strip() or not case.drug_labels:
            raise ValueError('患者对话和实际用药不能为空')
        self.case, self.dataset_path = case, dataset_path

    def clear_case(self) -> None:
        if self.vector_sender is not None:
            self.vector_sender.release()
        self.query_binary = None
        self.vector_name = None
        self.case, self.dataset_path = None, ''

    def provide_case_truth(self, _request: str) -> str:
        self.receive_task(State.IDLE)
        if self.case is None:
            raise RuntimeError('尚未开始病例')
        response = json.dumps({
            'case_id': self.case.case_id, 'patient_dialog': self.case.patient_input,
            'actual_drugs': list(self.case.drug_labels),
        }, ensure_ascii=False)
        self.transition(State.IDLE)
        return response

    def provide_recommendation_context(self, _request: str) -> str:
        """Provide a training case without labels, history or query embeddings."""
        self.receive_task(State.IDLE)
        if self.case is None:
            raise RuntimeError('尚未开始病例')
        data = {
            'case_id': self.case.case_id, 'dataset_path': self.dataset_path,
            'patient_dialog': self.case.patient_input, 'count': self.case.drug_count,
        }
        self.transition(State.IDLE)
        return json.dumps(data, ensure_ascii=False)

    def request_history(self, _request: str) -> str:
        self.receive_task(State.IDLE)
        if self.case is None:
            raise RuntimeError('尚未开始病例')
        # Text and vector state transfer are deliberately mutually exclusive.
        # The orchestrator keeps the dialogue for the later Doctor handoff;
        # Retrieval only needs it in text mode to create a query vector.
        data = {
            'case_id': self.case.case_id, 'dataset_path': self.dataset_path,
            'count': self.case.drug_count,
        }
        if self.vector_sender is not None:
            assert self.runtime is not None
            descriptor, self.query_binary = self.vector_sender.prepare(
                self.case.patient_input, self.runtime.trace_id, self.vector_name)
            data['query_embedding'] = descriptor
        else:
            data['patient_dialog'] = self.case.patient_input
        response = json.dumps(data, ensure_ascii=False)
        self.transition(State.IDLE)
        return response
