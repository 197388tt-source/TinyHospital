"""Tiny Hospital: sequential drug recommendation and experience mining."""

from __future__ import annotations

import argparse
import json
import math
import hashlib
import os
from pathlib import Path

from agent import Agent, DoctorAgent, ExpertAgent, PatientAgent, RetrievalAgent, State
from cases import (DIALMED_TEST_PATH, DIALMED_TRAIN_PATH, DialMedCase,
                   default_dataset_path, load_dialmed_cases,
                   memory_source_dataset_path)
from history.memory import CaseMemoryStore, GOLDEN_EXPERIENCE_PATH, REFLECTION_MEMORY_PATH
from tool.llm_client import LLMClient
from tool.local_qwen_client import DEFAULT_MODEL_PATH


def _restore_vector_history(case: DialMedCase, dataset_path: str, text: str) -> str:
    """Join a vector-only Retrieval reply with dialogue held by the orchestrator."""
    reply = json.loads(text)
    history = reply.get('history') if isinstance(reply, dict) else None
    if (not isinstance(reply, dict) or set(reply) != {'case_id', 'history'}
            or reply['case_id'] != case.case_id or not isinstance(history, dict)
            or set(history) != {'golden_experiences', 'reflections'}
            or any(not isinstance(values, list)
                   or any(not isinstance(value, str) or not value.strip() for value in values)
                   for values in history.values())):
        raise ValueError('向量检索回复无效或病例编号不匹配')
    return json.dumps({
        'case_id': case.case_id, 'dataset_path': str(dataset_path),
        'patient_dialog': case.patient_input, 'count': case.drug_count,
        'history': history,
    }, ensure_ascii=False)
from tool.llm_usage import append_llm_usage
from tool.diagnostics import emit
from tool.medicine_store import MedicineKnowledgeBase
from protocol import AgentPacket, MessageType, Runtime
from tool.tracing import Trace, append_communication_records


COMMUNICATION_USAGE_PATH = Path(__file__).resolve().parent / "runs/token_usage.jsonl"
CONTROL_USAGE_PATH = Path(__file__).resolve().parent / "runs/control_communication.jsonl"
LLM_USAGE_PATH = Path(__file__).resolve().parent / "runs/llm_usage.jsonl"


class HospitalSession:
    """A local TH-ACP runtime serving sequential DialMed cases."""

    def __init__(self, verbose: bool = True, golden_path: Path | None = None,
                 communication_path: Path | None = None, control_path: Path | None = None,
                 llm_path: Path | None = None, reflection_path: Path | None = None,
                 dataset_path: Path | None = None,
                 training: bool = True, retrieval_mode: str = 'embedding',
                 embedding_options: dict | None = None, vector_transport: str = 'shared_memory',
                 chat_timeout: float = 180.0, doctor_backend: str = 'api',
                 doctor_model_path: Path = DEFAULT_MODEL_PATH,
                 doctor_max_new_tokens: int = 256) -> None:
        if doctor_backend not in ('api', 'local-qwen'):
            raise ValueError('doctor_backend 必须为 api 或 local-qwen')
        if vector_transport not in ('text', 'binary', 'shared_memory'):
            raise ValueError('vector_transport 必须为 text、binary 或 shared_memory')
        dataset_path = Path(dataset_path) if dataset_path is not None else default_dataset_path(training)
        if retrieval_mode not in ('embedding', 'jaccard', 'none'):
            raise ValueError('retrieval_mode 必须为 embedding、jaccard 或 none')
        if training:
            retrieval_mode = 'none'
        self.vector_transport = vector_transport if retrieval_mode == 'embedding' else 'text'
        if self.vector_transport == 'binary':
            raise ValueError('binary 向量传输需要 process Runtime；sync 请用 shared_memory 或 text')
        if not math.isfinite(chat_timeout) or chat_timeout <= 0:
            raise ValueError('chat_timeout 必须为有限正数')
        self.trace = Trace(verbose=verbose)
        self.training = training
        self.dataset_path = str(dataset_path.resolve())
        self.communication_path = communication_path or COMMUNICATION_USAGE_PATH
        self.control_path = control_path or CONTROL_USAGE_PATH
        self.llm_path = llm_path or LLM_USAGE_PATH
        embedding_options = dict(embedding_options or {})
        embedding_options.setdefault('diagnostics_path', self.llm_path.parent / 'diagnostics.jsonl')
        embedding_options.setdefault('diagnostics_verbose', verbose)
        self.medicines = MedicineKnowledgeBase()
        memory_dataset = memory_source_dataset_path(dataset_path, training)
        self.memories = CaseMemoryStore(golden_path or GOLDEN_EXPERIENCE_PATH,
                                        reflection_path or REFLECTION_MEMORY_PATH,
                                        training, memory_dataset, retrieval_mode, embedding_options)
        self.patient = PatientAgent(self.trace, self.vector_transport, embedding_options)
        self.retrieval = RetrievalAgent(self.trace, self.memories, self.medicines,
                                        vector_transport=self.vector_transport)
        if doctor_backend == 'local-qwen':
            from agent.doctor_Qwen import DoctorAgent as QwenDoctorAgent
            from tool.local_qwen_client import LocalQwenClient
            self.doctor = QwenDoctorAgent(
                self.trace, self.medicines,
                lambda: LocalQwenClient(doctor_model_path, doctor_max_new_tokens, chat_timeout),
                training=training,
            )
        else:
            self.doctor = DoctorAgent(self.trace, self.medicines, LLMClient, training=training)
        self.expert = ExpertAgent(self.trace, self.memories, self.medicines, self.dataset_path, LLMClient)
        for model_agent in (self.doctor, self.expert):
            if hasattr(model_agent.llm, 'timeout'):
                model_agent.llm.timeout = chat_timeout
            model_agent.diagnostics_path = embedding_options['diagnostics_path']
            model_agent.diagnostics_verbose = embedding_options['diagnostics_verbose']
        self.agents = (self.patient, self.retrieval, self.doctor, self.expert)
        self.runtime = Runtime(self.trace)
        self._closed = False
        self.runtime.register_agent(self.patient, {
            'patient.provide_case_truth': self.patient.provide_case_truth,
            'patient.request_history': self.patient.request_history,
            'patient.provide_recommendation_context': self.patient.provide_recommendation_context,
        })
        self.runtime.register_agent(self.retrieval, {
            'retrieval.history': self.retrieval.retrieve_history,
            'retrieval.medicines': self.retrieval.retrieve_medicines,
        })
        self.runtime.register_agent(self.doctor, {'doctor.recommend': self.doctor.recommend})
        self.runtime.register_agent(self.expert, {
            'expert.receive_case_truth': self.expert.receive_case_truth,
            'expert.evaluate': self.expert.evaluate,
            'expert.review': self.expert.review,
        })
        append_communication_records(self.control_path, self.trace.packet_costs)

    def _system_request(self, capability: str, text: str):
        assert self.runtime.trace_id is not None
        delivery = self.runtime.deliver(AgentPacket.create(
            trace_id=self.runtime.trace_id, message_type=MessageType.REQUEST,
            source='System', payload={'capability': capability, 'text': text},
        ))
        if delivery.packet.head.message_type == MessageType.ERROR.value:
            raise RuntimeError(delivery.packet.payload['error'])
        return delivery

    def run_case(self, case: DialMedCase) -> dict:
        if self._closed or any(agent.state != State.IDLE for agent in self.agents):
            raise RuntimeError('会话已经结束或上一病例尚未完成')
        if not isinstance(case, DialMedCase):
            raise TypeError('必须提供 DialMedCase')
        trace_id = self.runtime.start_case()
        self.trace.begin_case(trace_id)
        self.patient.begin_case(case, self.dataset_path)
        truth = self._system_request('patient.provide_case_truth', '请将病例标签私下交给专家。')
        self.patient.send_request(truth.output, 'expert.receive_case_truth', truth.packet.head.message_id)
        if self.training:
            request = self._system_request('patient.provide_recommendation_context', '请提供对话和目标药物数量。')
            recommendation = self.patient.send_request(request.output, 'doctor.recommend',
                                                       request.packet.head.message_id)
        else:
            request = self._system_request('patient.request_history', '请提供对话和目标药物数量。')
            history = self.patient.send_request(request.output, 'retrieval.history', request.packet.head.message_id)
            history_text = history.output
            if self.vector_transport != 'text':
                history_text = _restore_vector_history(case, self.dataset_path, history_text)
            recommendation = self.retrieval.send_response(history_text, history.packet.head.message_id,
                                                           'doctor.recommend')
        recommendation_data = json.loads(recommendation.output)
        failed = recommendation_data.get('status') == 'doctor_failed'
        if failed:
            outcome = {**recommendation_data, 'actual_drugs': list(case.drug_labels),
                       'exact_match': False, 'reflection': None, 'golden_experience': None}
            emit(self.llm_path.parent / 'doctor_failures.jsonl', outcome)
            completed = recommendation
            completion_agent = self.doctor
            completion_message = '医生连续三次输出无效，已记录失败并跳过专家评审。'
            print(f"[病例跳过] 病例 {case.case_id}：{outcome['error']}；继续下一例。", flush=True)
        else:
            evaluation = self.doctor.send_request(recommendation.output, 'expert.evaluate',
                                                  recommendation.packet.head.message_id)
            completed = evaluation
            if self.training:
                introductions = self.expert.send_request(evaluation.output, 'retrieval.medicines',
                                                         evaluation.packet.head.message_id)
                completed = self.retrieval.send_response(introductions.output, introductions.packet.head.message_id,
                                                         'expert.review')
            outcome = json.loads(completed.output)
            completion_agent = self.expert
            completion_message = '病例评判与经验提炼已完成。' if self.training else '病例评判已完成。'
        if outcome.get('reflection_error'):
            print(f"[反思未保存] 病例 {case.case_id}：{outcome['reflection_error']}；保留评判结果，继续运行。", flush=True)
        append_llm_usage(self.llm_path, case.case_id,
                         self.doctor.model_calls + self.expert.model_calls)
        completion_agent.send_response(completion_message, completed.packet.head.message_id,
                                       'case.complete')
        if any(agent.state != State.IDLE for agent in self.agents):
            raise RuntimeError('病例结束后 Agent 未回到 IDLE')
        assert self.trace._case_packet_start is not None
        case_packets = self.trace.packet_costs[self.trace._case_packet_start:]
        self.trace.finish_case(trace_id)
        append_communication_records(self.communication_path, case_packets, case.case_id)
        self.patient.clear_case()
        self.doctor.clear_case()
        self.expert.clear_case()
        self.runtime.finish_case()
        if self.trace.verbose and not failed:
            kind = '完全匹配' if outcome['exact_match'] else '标签不匹配'
            print(f"病例 {case.case_id}：{kind}，推荐 {outcome['recommended_drugs']}")
            lesson = outcome['golden_experience'] or outcome['reflection']
            if lesson:
                print(f"经验：{lesson}")
        return outcome

    def close(self) -> None:
        if self._closed:
            return
        for agent in self.agents:
            if agent.state == State.ACTIVE:
                agent.transition(State.IDLE)
            if agent.state != State.STOPPED:
                agent.transition(State.STOPPED)
        if self.runtime.trace_id is not None:
            self.runtime.finish_case()
        self.patient.clear_case()
        self._closed = True
        if self.trace.verbose:
            self.trace.print_overall_summary()
            print('历史检索统计：' + json.dumps(self.retrieval_statistics, ensure_ascii=False))

    @property
    def retrieval_statistics(self):
        return {**self.memories.retrieval_statistics,
                'vector_transport': self.vector_transport,
                'vector_receive': dict(self.retrieval.vector_transport_stats),
                'query_sender': (dict(self.patient.vector_sender.stats)
                                 if self.patient.vector_sender is not None else None)}


def run_cases(cases: list[DialMedCase], verbose: bool = True,
              golden_path: Path | None = None, communication_path: Path | None = None,
              control_path: Path | None = None, llm_path: Path | None = None,
              reflection_path: Path | None = None,
              dataset_path: Path | None = None,
              training: bool = True, runtime_mode: str = 'sync',
              shared_dialog: bool = False,
              incremental_history: bool = False,
              incremental_medicines: bool = False,
              shared_medicines: bool = False,
              structured_data: bool = False, retrieval_mode: str = 'embedding',
              embedding_options: dict | None = None, vector_transport: str = 'shared_memory',
              chat_timeout: float = 180.0, agent_timeout: float = 720.0,
              on_case_complete=None, doctor_backend: str = 'api',
              doctor_model_path: Path = DEFAULT_MODEL_PATH,
              doctor_max_new_tokens: int = 256) -> tuple[list[dict], Trace]:
    if doctor_backend != 'api' and runtime_mode != 'sync':
        raise ValueError('本地 Qwen Doctor 请使用 --runtime sync')
    if not cases:
        raise ValueError('至少需要一个病例')
    dataset_path = Path(dataset_path) if dataset_path is not None else default_dataset_path(training)
    if runtime_mode == 'sync':
        if (shared_dialog or incremental_history or incremental_medicines
                or shared_medicines or structured_data):
            raise ValueError('共享对话和增量通信需要 process Runtime')
        session_type = HospitalSession
    elif runtime_mode == 'process':
        from hospital_process import ProcessHospitalSession
        session_type = ProcessHospitalSession
    else:
        raise ValueError('runtime_mode 只支持 sync 或 process')
    options = {'timeout': agent_timeout, 'shared_dialog': shared_dialog,
               'incremental_history': incremental_history,
               'incremental_medicines': incremental_medicines,
               'shared_medicines': shared_medicines,
               'structured_data': structured_data} if runtime_mode == 'process' else {}
    if runtime_mode == 'sync':
        options.update(doctor_backend=doctor_backend, doctor_model_path=doctor_model_path,
                       doctor_max_new_tokens=doctor_max_new_tokens)
    session = session_type(verbose, golden_path, communication_path, control_path,
                           llm_path, reflection_path, dataset_path, training,
                           retrieval_mode=retrieval_mode, embedding_options=embedding_options,
                           vector_transport=vector_transport, chat_timeout=chat_timeout, **options)
    try:
        outcomes = []
        for case in cases:
            outcome = session.run_case(case)
            if on_case_complete is not None:
                on_case_complete(outcome)
            outcomes.append(outcome)
    finally:
        session.close()
    return outcomes, session.trace


def prepare_test_memories(source: Path, target: Path, *, resume: bool) -> None:
    """Copy selected memories once; never overwrite experiment snapshots."""
    names = ('golden_experience.json', 'reflection_memory.json')
    if resume:
        if not all((target / name).is_file() for name in names):
            raise ValueError('续跑缺少冻结经验文件')
        return
    contents = {}
    for name in names:
        raw = (source / name).read_bytes()
        rows = json.loads(raw)
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError(f'经验文件格式错误：{source / name}')
        destination = target / name
        if destination.exists() and destination.read_bytes() != raw:
            raise ValueError(f'实验目录存在不同的经验文件：{destination}；请使用新目录')
        contents[name] = raw
    target.mkdir(parents=True, exist_ok=True)
    for name, raw in contents.items():
        try:
            with (target / name).open('xb') as stream:
                stream.write(raw)
        except FileExistsError:
            if (target / name).read_bytes() != raw:
                raise ValueError(f'经验文件已变化：{target / name}')


def main() -> None:
    parser = argparse.ArgumentParser(description='按 DIALMED 顺序执行 Tiny Hospital 药物推荐模拟')
    parser.add_argument('--count', type=int, default=3, help='从数据集开头连续读取的病例数')
    parser.add_argument('--dataset', type=Path, default=None,
                        help='DIALMED JSONL 路径；默认训练用 traincut.txt，测试用 testcut.txt')
    parser.add_argument('--doctor-backend', choices=('api', 'local-qwen'), default='api')
    parser.add_argument('--doctor-model-path', type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument('--doctor-max-new-tokens', type=int, default=256)
    parser.add_argument('--memory-dir', type=Path, default=None,
                        help='测试经验来源目录；首次运行复制到实验目录，续跑使用冻结副本；本地 Qwen 默认 history')
    parser.add_argument('--stage', choices=('train', 'test'), default='train', help='训练可写记忆；测试使用冻结快照')
    parser.add_argument('--experiment-dir', type=Path, default=None,
                        help='实验目录：保存独立经验、向量索引和日志；相对路径基于当前工作目录')
    parser.add_argument('--retrieval-mode', choices=('embedding', 'jaccard', 'none'), default='embedding')
    parser.add_argument('--vector-transport', choices=('text', 'binary', 'shared_memory'),
                        default='shared_memory', help='Patient 到 Retrieval 的查询载荷方式')
    parser.add_argument('--embedding-backend', choices=('api', 'local'), default='api')
    parser.add_argument('--embedding-model', default='Qwen3-Embedding-8B')
    parser.add_argument('--embedding-model-version', default=None)
    parser.add_argument('--embedding-endpoint', default='http://aigw.dlut.edu.cn/v1/embeddings')
    parser.add_argument('--embedding-store', type=Path, default=None)
    parser.add_argument('--embedding-key-env', default='EMBEDDING_API_KEY')
    parser.add_argument('--embedding-timeout', type=float, default=120)
    parser.add_argument('--chat-timeout', type=float, default=180,
                        help='API 请求超时或本地生成时间上限秒数；本地在生成步骤结束后检查，不含加载')
    parser.add_argument('--agent-timeout', type=float, default=720, help='单次 Agent 任务总等待上限（含内容重试、记忆同步）')
    parser.add_argument('--embedding-min-similarity', type=float, default=0.60,
                        help='embedding 余弦相似度下限；0.60 是待校准的保守初值')
    parser.add_argument('--runtime', choices=('sync', 'process'), default='sync',
                        help='同步 Runtime 或四 Agent 独立进程')
    parser.add_argument('--shared-dialog', action='store_true',
                        help='在 process Runtime 中复用 patient_dialog 共享内存')
    parser.add_argument('--incremental-history', action='store_true',
                        help='在 process Runtime 中只传递 Retrieval 新增的历史经验')
    parser.add_argument('--incremental-medicines', action='store_true',
                        help='在 process Runtime 中只传递 Retrieval 新增的药物说明')
    parser.add_argument('--shared-medicines', action='store_true',
                        help='Retrieval 经共享内存向 Expert 传递药物说明')
    parser.add_argument('--structured-data', action='store_true',
                        help='在 process Runtime 中将业务 JSON 对象放入 TH-ACP payload.data')
    parser.add_argument('--resume', action='store_true', help='从实验目录跳过已完成病例')
    parser.add_argument('--allow-code-change', action='store_true',
                        help='配合 --resume 接受代码变更并留档；数据和运行配置仍须一致')
    parser.add_argument('--api-retries', type=int, default=2, help='临时 API 错误额外重试次数，0 到 5')
    args = parser.parse_args()
    if not 0 <= args.api_retries <= 5:
        parser.error('--api-retries 必须为 0 到 5')
    if args.resume and args.experiment_dir is None:
        parser.error('--resume 需要 --experiment-dir')
    if args.allow_code_change and not args.resume:
        parser.error('--allow-code-change 必须配合 --resume')
    if args.doctor_max_new_tokens < 1:
        parser.error('--doctor-max-new-tokens 必须大于 0')
    if args.doctor_backend == 'local-qwen':
        if args.runtime != 'sync':
            parser.error('本地 Qwen Doctor 需要 --runtime sync')
        if args.stage == 'test' and args.memory_dir is None:
            args.memory_dir = GOLDEN_EXPERIENCE_PATH.parent
        args.doctor_model_path = args.doctor_model_path.expanduser().resolve()
    if args.memory_dir is not None:
        if args.stage != 'test' or args.experiment_dir is None:
            parser.error('--memory-dir 需要 --stage test 和独立的 --experiment-dir')
        args.memory_dir = args.memory_dir.expanduser().resolve()
    os.environ['TINY_HOSPITAL_API_RETRIES'] = str(args.api_retries)
    training = args.stage == 'train'
    dataset_path = args.dataset or default_dataset_path(training)
    if any(not math.isfinite(x) or x <= 0 for x in (args.chat_timeout, args.agent_timeout, args.embedding_timeout)):
        parser.error('所有 timeout 必须为有限正数')
    if args.runtime == 'process' and args.agent_timeout <= max(args.chat_timeout, args.embedding_timeout):
        parser.error('--agent-timeout 必须大于 --chat-timeout 和 --embedding-timeout')
    if args.count < 1:
        parser.error('--count 必须大于 0')
    if not 0 <= args.embedding_min_similarity <= 1:
        parser.error('--embedding-min-similarity 必须在 0 到 1 之间')
    if args.shared_dialog and args.runtime != 'process':
        parser.error('--shared-dialog 需要 --runtime process')
    if args.incremental_history and args.runtime != 'process':
        parser.error('--incremental-history 需要 --runtime process')
    if args.incremental_medicines and args.runtime != 'process':
        parser.error('--incremental-medicines 需要 --runtime process')
    if args.shared_medicines and args.runtime != 'process':
        parser.error('--shared-medicines 需要 --runtime process')
    if args.structured_data and args.runtime != 'process':
        parser.error('--structured-data 需要 --runtime process')
    from embedding import EmbeddingConfig
    embedding_options = {
        'config': EmbeddingConfig(args.embedding_backend, args.embedding_model,
                                  args.embedding_model_version or args.embedding_model,
                                  endpoint=args.embedding_endpoint),
        'api_key_env': args.embedding_key_env, 'timeout': args.embedding_timeout,
        'path': args.embedding_store, 'min_similarity': args.embedding_min_similarity,
    }
    cases = load_dialmed_cases(dataset_path, args.count)
    experiment_paths = {}
    if args.experiment_dir is not None:
        experiment_dir = args.experiment_dir.expanduser().resolve()
        golden_path = experiment_dir / 'golden_experience.json'
        reflection_path = experiment_dir / 'reflection_memory.json'
        if args.memory_dir is not None:
            if args.memory_dir == experiment_dir:
                parser.error('--experiment-dir 必须与经验来源目录不同')
            try:
                prepare_test_memories(args.memory_dir, experiment_dir, resume=args.resume)
            except (ValueError, OSError) as exc:
                parser.error(str(exc))
        if not training and not all(path.is_file() for path in (golden_path, reflection_path)):
            parser.error('实验目录缺少经验文件，请先使用同一 --experiment-dir 运行训练')
        experiment_dir.mkdir(parents=True, exist_ok=True)
        if training:
            for path in (golden_path, reflection_path):
                try:
                    with path.open('x', encoding='utf-8') as stream:
                        stream.write('[]\n')
                except FileExistsError:
                    pass
        log_dir = experiment_dir / args.stage
        log_dir.mkdir(parents=True, exist_ok=True)
        experiment_paths = {
            'golden_path': golden_path,
            'reflection_path': reflection_path,
            'communication_path': log_dir / 'token_usage.jsonl',
            'control_path': log_dir / 'control_communication.jsonl',
            'llm_path': log_dir / 'llm_usage.jsonl',
        }
        embedding_options['path'] = args.embedding_store or experiment_dir / 'semantic-memory.sqlite'
        embedding_options['diagnostics_path'] = log_dir / 'diagnostics.jsonl'
        print(f'实验目录：{experiment_dir}', flush=True)
    progress = None
    if args.experiment_dir is not None:
        from tool.experiment_progress import ExperimentProgress
        from tool.llm_client import DEFAULT_MODEL, DEFAULT_BASE_URL
        source_root = Path(__file__).resolve().parent
        config = {key: str(value) if isinstance(value, Path) else value
                  for key, value in vars(args).items()
                  if key not in ('resume', 'allow_code_change', 'count', 'api_retries', 'experiment_dir')}
        config['dataset_sha256'] = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
        config['memory_source_sha256'] = hashlib.sha256(
            memory_source_dataset_path(dataset_path, training).read_bytes()).hexdigest()
        config['chat_model'] = DEFAULT_MODEL
        config['chat_endpoint'] = DEFAULT_BASE_URL
        if args.doctor_backend == 'local-qwen':
            config['chat_model'] = str(args.doctor_model_path)
            config['chat_endpoint'] = 'local'
            config['doctor_generation'] = {'enable_thinking': False, 'do_sample': False,
                                           'max_new_tokens': args.doctor_max_new_tokens}
        config['expert_model'] = DEFAULT_MODEL
        config['expert_endpoint'] = DEFAULT_BASE_URL
        config['code_sha256'] = {str(path.relative_to(source_root)):
                                hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in sorted(source_root.rglob('*.py'))
                                if path.parent == source_root or path.parts[-2] in
                                ('agent', 'tool', 'history', 'embedding')}
        try:
            progress = ExperimentProgress(experiment_dir, args.stage, config, args.resume,
                                          allow_code_change=args.allow_code_change)
        except (ValueError, BlockingIOError) as exc:
            parser.error(str(exc))
        completed = set(progress.state['completed_case_ids'])
        cases = [case for case in cases if case.case_id not in completed]
        experiment_paths['on_case_complete'] = progress.commit
    try:
        if not cases:
            print('指定范围内的病例均已完成，无需调用 API。')
            return
        run_cases(cases, dataset_path=dataset_path,
                  **experiment_paths,
                  retrieval_mode=args.retrieval_mode, embedding_options=embedding_options,
                  vector_transport=args.vector_transport,
                  chat_timeout=args.chat_timeout, agent_timeout=args.agent_timeout,
                  training=training, runtime_mode=args.runtime,
                  shared_dialog=args.shared_dialog,
                  incremental_history=args.incremental_history,
                  incremental_medicines=args.incremental_medicines,
                  shared_medicines=args.shared_medicines,
                  structured_data=args.structured_data,
                  doctor_backend=args.doctor_backend, doctor_model_path=args.doctor_model_path,
                  doctor_max_new_tokens=args.doctor_max_new_tokens)
    finally:
        if progress is not None:
            progress.close()


if __name__ == '__main__':
    main()
