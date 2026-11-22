"""Agent state machine, packet helpers, and shared model client behavior."""

from __future__ import annotations

from enum import Enum
from time import perf_counter

from tool.diagnostics import emit
from typing import Callable

from tool.llm_client import LLMClient
from protocol import AgentPacket, MessageType, Runtime
from tool.tracing import Trace


class State(str, Enum):
    IDLE = "IDLE"
    ACTIVE = "ACTIVE"
    WAITING = "WAITING"
    STOPPED = "STOPPED"


ALLOWED_TRANSITIONS = {
    State.IDLE: {State.ACTIVE, State.STOPPED},
    State.ACTIVE: {State.IDLE, State.WAITING},
    State.WAITING: {State.ACTIVE, State.STOPPED},
    State.STOPPED: set(),
}


class Agent:
    def __init__(self, name: str, trace: Trace) -> None:
        self.name = name
        self.trace = trace
        self.runtime: Runtime | None = None
        self.state = State.IDLE
        self.trace.state(name, "INIT", State.IDLE.value)

    def transition(self, target: State) -> None:
        if target not in ALLOWED_TRANSITIONS[self.state]:
            raise RuntimeError(f"{self.name} 不允许 {self.state.value} -> {target.value}")
        before = self.state
        self.state = target
        self.trace.state(self.name, before.value, target.value)

    def receive_task(self, expected: State) -> None:
        if self.state != expected:
            raise RuntimeError(f"{self.name} 当前为 {self.state.value}，预期 {expected.value}")
        self.transition(State.ACTIVE)

    def make_packet(
        self,
        message_type: MessageType,
        *,
        payload: dict | None = None,
        correlation_id: int | None = None,
    ) -> AgentPacket:
        if self.runtime is None:
            raise RuntimeError(f"{self.name} 尚未连接 Runtime")
        trace_id = 0 if message_type == MessageType.HELLO else self.runtime.trace_id
        if trace_id is None:
            raise RuntimeError("当前没有运行中的病例")
        return AgentPacket.create(
            trace_id=trace_id, message_type=message_type,
            source=self.name, payload=payload, correlation_id=correlation_id,
        )

    def send_request(
        self, text: str, capability: str, correlation_id: int | None = None,
    ):
        assert self.runtime is not None
        delivery = self.runtime.deliver(self.make_packet(
            MessageType.REQUEST, payload={"capability": capability, "text": text},
            correlation_id=correlation_id,
        ))
        if delivery.packet.head.message_type == MessageType.ERROR.value:
            raise RuntimeError(delivery.packet.payload["error"])
        return delivery

    def send_parameters(self, capability: str, parameters: dict, correlation_id: int):
        assert self.runtime is not None
        delivery = self.runtime.deliver(self.make_packet(
            MessageType.REQUEST,
            payload={"capability": capability, "parameters": parameters},
            correlation_id=correlation_id,
        ))
        if delivery.packet.head.message_type == MessageType.ERROR.value:
            raise RuntimeError(delivery.packet.payload["error"])
        return delivery

    def send_response(
        self, text: str, correlation_id: int, capability: str,
    ):
        assert self.runtime is not None
        delivery = self.runtime.deliver(self.make_packet(
            MessageType.RESPONSE, payload={"capability": capability, "text": text},
            correlation_id=correlation_id,
        ))
        if delivery.packet.head.message_type == MessageType.ERROR.value:
            raise RuntimeError(delivery.packet.payload["error"])
        return delivery


class ModelAgent(Agent):
    def __init__(self, name: str, trace: Trace,
                 llm_client_factory: Callable[[], LLMClient] = LLMClient) -> None:
        self.llm = llm_client_factory()
        super().__init__(name, trace)
        self.model_calls: list[dict] = []

    def clear_model_calls(self) -> None:
        self.model_calls = []

    def ask_llm(self, step: str, instruction: str, context: str) -> str:
        memory = getattr(self, 'memories', None)
        path = getattr(self, 'diagnostics_path', getattr(memory, 'diagnostics_path', None))
        verbose = getattr(self, 'diagnostics_verbose', getattr(memory, 'diagnostics_verbose', False))
        case = getattr(self, 'case', None) or {}
        row = {
            'agent': self.name, 'step': step,
            'attempt': 1 + sum(call['step'] == step for call in self.model_calls),
            'flow_step': self.trace.message_costs[-1].step,
            'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0,
            'usage_known': False, 'request_timeout_seconds': getattr(self.llm, 'timeout', None),
            'status': 'api_error',
        }
        identity = {'case_id': case.get('case_id', getattr(self, 'case_id', None)),
                    'trace_id': getattr(self.runtime, 'trace_id', None)}
        emit(path, {'event': 'model_call_started', **identity, **row})
        if verbose:
            print(f"[模型等待] {self.name} 病例 {identity['case_id']} "
                  f"{step} 第{row['attempt']}次调用开始。", flush=True)
        started = perf_counter()
        try:
            reply = self.llm.chat([
                {'role': 'system', 'content': instruction},
                {'role': 'user', 'content': context},
            ])
        except Exception as exc:
            row.update(status='timeout' if isinstance(exc, TimeoutError) else 'api_error',
                       error_type=type(exc).__name__)
            raise
        else:
            usage = self.llm.last_usage
            row.update(status='returned', usage_known=True,
                       prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens,
                       total_tokens=usage.total_tokens)
            return reply
        finally:
            row['elapsed_ms'] = round((perf_counter() - started) * 1000, 3)
            self.model_calls.append(row)
            emit(path, {'event': 'model_call_finished', **identity, **row})
            if verbose:
                print(f"[模型返回] {self.name} {step} 状态={row['status']} "
                      f"耗时={row['elapsed_ms']:.1f} ms", flush=True)
