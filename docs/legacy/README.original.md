> 历史文档：保留设计演进资料，命令、路径、模型及行为可能已过时。当前运行以根目录 README.md 和源码为准。

# Tiny Hospital：DIALMED 药物推荐与经验提炼

项目使用自己的 TH-ACP 协议、本地 Runtime、Agent 和 JSON 记忆文件；运行时不导入 YuLan-OneSim。

本地 Qwen Doctor 连续三次输出无效时，会记录 `status=doctor_failed` 并继续下一病例，
不调用 Expert、不写入经验库。三次原始回复和错误保存到实验目录的
`train/doctor_failures.jsonl`；`train/progress.json` 也保存失败结果，并在续跑时跳过这些已处理病例。
无实验目录时，失败日志位于 `runs/doctor_failures.jsonl`。模型加载、运行故障和 API 异常仍会报错停止。

更新代码后若需接着已有进度运行，可使用 `bash train_qwen.sh --resume --allow-code-change`。
该选项仅允许代码哈希变化，在进度文件的 `code_changes` 留档；数据、模型与运行配置仍必须一致。
首次运行不用这些参数，之后代码未变时只需 `--resume`。

## 按实验保存经验

在 `tiny_hospital` 目录运行，训练和测试指定同一个实验目录：

```bash
python main.py --stage train --count 686 --experiment-dir experiment/expert1
python main.py --stage test --count 93 --experiment-dir experiment/expert1
```

目录内保存 `golden_experience.json`、`reflection_memory.json` 和默认向量库
`semantic-memory.sqlite`；通信、模型用量及诊断日志分别保存到 `train/`、`test/`。
首次训练只创建缺失的空经验文件，不复制 `history/` 的已有经验。
测试要求两份经验文件已经存在，读取冻结快照，不修改经验 JSON。
测试阶段只检索已有经验、调用 Doctor 推荐并由 Expert 在本地比较标签；
不查询药物说明，不调用 Expert 模型生成成功经验或错误反思。
测试结果中的 `golden_experience` 和 `reflection` 均为 `null`。
同步和多进程 Runtime 均遵循这一规则；训练阶段仍执行完整经验提炼流程。
相对路径以当前工作目录为基准；也可传入绝对路径。
显式传入 `--embedding-store` 时，该路径优先于实验目录内的默认向量库。

不同实验应使用不同目录，例如修改 Expert 后使用 `experiment/expert2`。
已有进度时必须加 `--resume`，程序跳过已完成病例；日志仍追加。
训练和测试分别保存进度。
不传 `--experiment-dir` 时仍使用原来的 `history/` 和 `runs/` 路径。
每阶段的 `progress.json` 原子保存 `completed_case_ids`、逐例 `results`、经验快照和配置。
结果包含推荐药物、标签、完全匹配判断及经验/反思；尚未增加汇总指标。

### 网关故障重试与续跑

```bash
# 首次运行：临时网络错误最多额外重试 2 次（最多请求 3 次）
python -u main.py --stage train --count 686 --experiment-dir experiment/expert1 --api-retries 2
# 中断后恢复：count 仍是数据集开头的目标总数，不是新增病例数
python -u main.py --stage train --count 686 --experiment-dir experiment/expert1 --api-retries 2 --resume
# 测试有独立进度；测试中断后同样追加 --resume
python -u main.py --stage test --count 93 --experiment-dir experiment/expert1 --api-retries 2
```

聊天与 embedding 请求均重试网络连接错误、超时及 HTTP 408/429/500/502/503/504，
按 2、4、8、16、30 秒退避；401/403 等认证错误不重试。`--api-retries` 范围为 0–5。
这是每次网络请求的重试上限；原有模型内容校验重试独立存在。
网络超时不代表服务端未执行，重复请求可能产生额外用量。
进程模式的 `--agent-timeout` 是包含重试的总任务时限，需要留出足够时间。

每例完整返回后才提交进度。恢复训练时，恢复到最后一次提交的经验快照，
重新执行未完成病例；语义索引由已有同步逻辑更新。API 已发生的调用和旧日志不会回滚。
Expert 按原逻辑返回 `reflection_error` 的病例仍属于已完成，错误信息保存在结果中。
恢复会校验代码、数据和配置；修改 Expert 后应新建实验目录。
同一实验目录只允许一个训练或测试进程运行。不要在运行中手动编辑经验文件；
恢复训练会用已提交快照替换经验 JSON。历史实验若没有 `progress.json`，不能推断已完成病例，
应使用新目录开始。

```text
训练：System → Patient → Retrieval（保留路由，历史不进入 Doctor 模型）→ Doctor（推荐药物与依据）
       → Expert（与标签比较）→ Retrieval（药物说明）
       → Expert（提炼正向经验或错误反思）→ End

测试：System → Patient → Retrieval（已有经验与反思）→ Doctor（推荐药物）
       → Expert（本地标签比较，不调用模型）→ End
```

Patient 把完整医患对话和真实药物标签私下交给 Expert；Doctor 看不到当前病例真实药名。
Doctor 按 `--stage` 使用两套提示词：训练阶段模型只接收当前对话、目标药物数量和药物白名单，
不接收历史经验，输出包含 `recommended_drugs` 和 `reasoning_summary` 的 JSON 对象，供 Expert 分析。
现有 Retrieval 路由和索引更新仍保留，但训练时检索到的历史内容不会进入 Doctor 的系统提示词、用户输入或重试输入。
测试阶段根据检索结果选择提示词：正向经验或错误反思任一非空时，使用现有带经验的提示词；
两类均为空（或未提供历史经验）时，使用与 `base_test/base.py` 完全相同的基线推荐指令，用户输入仅包含当前医患对话。
模型仅输出药名 JSON 数组，不生成或校验判断依据；内部消息封装为 `case_id` 和 `recommended_drugs` 后本地评分。
两类历史经验均允许为空，经验不得被当作当前对话原文；使用前须核对病例条件。
两阶段仍沿用现有目标药物数量、药名去重约束和内容重试策略。

训练阶段完全匹配时，Expert 额外调用一次 LLM 提炼不超过 120 字的正向选药经验。标签匹配不自动等于推理正确；只有对话原文、医生判断和药物适应症支持同一原则时，才写入 `history/golden_experience.json`。证据不足时，本例不入库。未完全匹配时，Expert 围绕多荐和漏荐药物分析选药依据、联用限制及相似药物取舍，提炼一条纠错反思：建议 120～200 个字符，最多 250 个字符，允许 1～3 句并写成一个段落。有效反思写入 `history/reflection_memory.json`；证据不足时模型返回 `{"reflection":""}`，记录为 `unsupported`，结果中的 `reflection` 为 `null`，不重试、不新增记忆。反思应区分有依据的推荐问题与单纯的标签差异。

记忆文件只保留两类：

- `golden_experience.json`：每条为 `case_id`、`experience`，不保存正确案例或真实药物标签。
- `reflection_memory.json`：每条仅为 `case_id`、`reflection` 两字段；不保存药物列表或病例摘要。

Retrieval 用 `case_id` 从训练集恢复经验或反思来源的对话，默认使用 embedding 余弦相似度排序（`--retrieval-mode embedding`）；可切换 `jaccard` 或 `none`。查询向量只编码当前对话，候选向量只编码历史来源对话；经验或反思文字不参与相似度计算，只在来源对话命中后返回。embedding 默认阈值为待校准的保守初值 0.60（`--embedding-min-similarity`），Jaccard 保持 0.15，两类记忆合计最多返回 3 条；排除当前病例和重复对话。传给 Doctor 的历史内容仅为 `golden_experiences` 和 `reflections` 两组文本数组，不含历史病例摘要、对话、编号、药物标签或相似度；这些信息仅用于内部检索。若训练集缺少某条记忆对应的对话，该记忆无法参与检索。训练阶段新经验立即更新索引；只修改经验文字时复用未变的来源对话向量。训练阶段也会检测外部文件更新；测试阶段冻结启动时的记忆快照且不写入。使用其他训练集时，应给它配置独立的经验和反思文件。

每例只返回一个内存中的简要结果，不生成病例报告，也不再写入 `runs/reports.json`。既有的 `runs/reports.json` 是旧运行记录，当前程序不会读取或改写它。通信估算继续写入 `runs/token_usage.jsonl`，模型实际 Token 用量写入 `runs/llm_usage.jsonl`，其中 `golden_experience_total` 单独统计正向经验提炼开销。命令行结束时会打印检索命中率、耗时、返回数量和相似度统计。

`runs/token_usage.jsonl` 的每条报文现在还包含 `step`、`execution_ns` 和 `serialization_ns`。执行耗时是同步 `Runtime.deliver` 总时间，包含 Handler 和真实运行时的模型 API 等待；序列化耗时包含 TH-ACP JSON 编码、解析与校验，以及 Trace 为统计再次编码的耗时，是执行耗时的一部分。`message_tokens_estimate` / `packet_tokens_estimate` 只是通信字符规则估算，不能与 `runs/llm_usage.jsonl` 的模型 API 实际 Token 相加或等同。

第一阶段的固定 10 例基线保存在 `runs/performance_baseline.json` 和 `runs/performance_baseline.md`。第二阶段将步骤 7 的两份说明映射改为 `recommended_drugs`、`actual_drugs` 与统一的 `medicine_introductions`；Expert 按药名顺序在本地还原原有两份映射。其余通信步骤、TH-ACP 报文结构和模型提示词保持原样。

下列命令使用与第一阶段相同的确定性模型桩、病例及冻结历史记忆，生成逐步骤及逐病例比较报告；临时运行日志不会写入项目的历史记忆文件。单次本地耗时不代表真实模型 API 延迟或 Token 消耗。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 performance_stage2.py
```

报告写入 `runs/performance_stage2.json` 和 `runs/performance_stage2.md`。`performance_baseline.py` 是共用测量夹具，单独运行时仅写入 `runs/performance_current.json`，不会覆盖第一阶段基线。

## 第三阶段：可选进程间通信

`Runtime` 的同步 Handler 调用仍是默认方式。`ipc_runtime.py` 提供 `MultiprocessRuntime`，对通过 `register_process_agent(name, capability, handler)` 明确注册的文本 `REQUEST` 和 `RESPONSE`，使用 `multiprocessing` 的 `spawn` 子进程和 `Pipe` 往返传递序列化 TH-ACP 报文。父进程统一分配返回报文的 `message_id`；子进程返回的 `RESPONSE` 或 `ERROR` 保留输入报文的 `trace_id`，并以输入报文的 `message_id` 作为 `correlation_id`。其余 capability 继续走同步 Runtime。当前接口只演示单报文在途的文本通信，不接入完整医疗 Agent 或结构化检索参数。

```python
from ipc_runtime import MultiprocessRuntime
from protocol import AgentPacket, MessageType
from tool.tracing import Trace

def echo(text: str) -> str:  # spawn 子进程需要可导入的顶层函数
    return text

if __name__ == "__main__":
    trace = Trace(verbose=False)
    with MultiprocessRuntime(trace, timeout=2.0) as runtime:
        runtime.register_process_agent("EchoAgent", "demo.echo", echo)
        case_id = runtime.start_case()
        trace.begin_case(case_id)
        request = AgentPacket.create(
            trace_id=case_id, message_type=MessageType.REQUEST,
            source="System", payload={"capability": "demo.echo", "text": "你好"},
        )
        delivery = runtime.deliver(request)
        assert delivery.output == "你好"
        assert delivery.packet.head.correlation_id == request.head.message_id
```

`register_process_agent` 等待子进程发出就绪信号后才注册 capability；`startup_timeout` 控制这段等待。Handler 异常返回关联的 TH-ACP `ERROR`；写入、读取或 Handler 超时以及子进程连接中断也返回 `ERROR`，并终止和回收对应子进程。`timeout` 是每次 IPC 往返的总截止时间。`close()` 或上下文管理器负责正常退出时的清理。调用方应检查 `delivery.packet.head.message_type`。使用顶层可 pickle 的确定性 Handler 测试，不调用 LLM。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_ipc_runtime -v
PYTHONDONTWRITEBYTECODE=1 python3 performance_ipc.py
```

对比报告写入 `runs/performance_ipc.json`，分别记录 `REQUEST` 和 `RESPONSE` 预热后的同步与 IPC 往返延迟、TH-ACP 报文字节数、Pipe 输入帧大小，以及子进程启动和就绪耗时。统计包含 Runtime 与 Trace 开销，单机数字仅用于本地性能基线。

## 第四阶段：四 Agent 跨进程协作

`hospital_process.py` 提供 `ProcessHospitalSession`。Patient、Retrieval、Doctor、Expert 在四个独立子进程中保持各自的病例状态；主进程按上文原有九步顺序生成 TH-ACP `REQUEST` 或 `RESPONSE`，通过 `MultiprocessRuntime` 投递。每个子进程可注册多个 capability。病例开始、结束和异常中止使用同一 Pipe 上的管理帧；这些管理帧只用于进程生命周期，不改变 TH-ACP 业务报文格式。子进程不再向 Runtime 发起嵌套请求。

```bash
python3 main.py --count 3 --stage test --runtime process
```

默认 `--runtime sync` 仍使用原同步流程。进程模式复用原 Agent Handler、提示词、药物逻辑和 `CaseMemoryStore`：Patient 在病例开始时接收原始标签，Expert 只在私有传递步骤接收标签；Doctor 的进程只收到预测所需对话、目标数量、白名单和历史经验。训练阶段由 Expert 写入既有记忆文件，Retrieval 下次检索按原逻辑刷新；测试阶段各进程固定启动时的记忆快照。Handler 错误会返回关联的 `ERROR`，病例中止时重置 Agent 状态，可继续运行下一例；子进程崩溃或通信失败会使当前会话不可继续，应关闭并重建会话。

不调用真实模型的完整流程回归：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_hospital_process -v
```

该测试使用确定性模型，验证同步和进程版连续 10 例结果一致、训练写入与下例检索、测试冻结、异常后继续运行，以及四个子进程退出清理。本阶段仍使用 Pipe，没有引入共享内存，也没有进行最终性能优化实验。

## 第五阶段：状态与异常恢复

进程会话的状态按 `STARTING → READY → RUNNING → READY` 转换。病例失败时先进入 `ABORTING`：如果只是 Handler 返回关联的 `ERROR`，四个 Agent 会清理病例数据并回到 `IDLE`，会话回到 `READY`，可继续处理其他病例。每个 Agent 的病例状态为 `IDLE → CASE_READY → EXECUTING → CASE_READY → IDLE`；Handler 抛错时转为 `FAILED`，随后由 `abort_case` 清理。同步 Runtime 的状态和流程不受影响。

IPC Runtime 在正常病例中按 `READY → CASE_ACTIVE → READY` 转换。子进程崩溃、通信超时、Pipe 断开或响应关联校验失败会使 Runtime 转为 `FAILED`，立即关闭 Pipe、终止并回收全部子进程；当前病例的错误交给会话。会话也转为 `FAILED`，需要调用 `close()`，然后创建新会话处理后续病例。`close()` 可重复调用，结束后状态为 `CLOSED`。失败病例 ID 在当前会话内禁止再次执行；系统不会自动重试病例，因为 Expert 可能已写入记忆而确认报文尚未到达。重建会话后若要手动重跑失败病例，应先检查已有记忆和输出记录。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_stage5_recovery -v
```

故障注入覆盖正常结束、Handler 异常后继续运行、超时、子进程崩溃、连接断开、关闭并重建会话，以及经验写入后发生错误时禁止同会话重试。本阶段仍使用 Pipe，不进行最终性能对照实验。

## 第六阶段：Doctor→Expert 二进制决策状态

进程模式下，Doctor 原有 `recommend` 方法仍产生完整推荐结果。进程边界使用 `medicine/medicine.json` 的固定编号，将 `recommended_drugs` 编为位向量，并附带编号顺序以保留推荐顺序。二进制块包含版本、目录指纹和长度校验；它作为独立的原始 bytes 附在同一 Pipe 帧中的 TH-ACP 报文后面，不进入 JSON 数组或 Base64 字符串。报文的 `text` 只保留 `case_id` 和原有 `reasoning_summary`。Expert 收到后校验并恢复药名列表，再调用未修改的原有 `evaluate` 方法。同步 Runtime 继续传递原完整 JSON。

二进制载荷仅允许用于 `doctor.recommend` 的响应和 `expert.evaluate` 的请求。无效向量由 Expert 返回关联的 TH-ACP `ERROR`，病例按第五阶段规则中止；传输帧或关联信息损坏则使会话失败并清理子进程。父进程记录这两次实际 Pipe 帧长度，Agent 记录编码和解码耗时。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_decision_vector tests.test_hospital_process -v
PYTHONDONTWRITEBYTECODE=1 python3 measure_decision_transfer.py
```

确定性模型运行 10 例的统计写入 `runs/stage6_decision_transfer.json`。统计只涵盖 Doctor 响应与 Expert 请求，JSON 方案按相同 TH-ACP 报文头重建，均计入 Pipe 帧头与请求响应 ID；不用于最终性能基线。其余医疗链路、记忆算法和文件格式未改动。

## 第七阶段：可选共享内存传输

`MultiprocessRuntime.deliver_shared(packet, fields=None, min_bytes=256)` 为进程 Agent 的单次请求选择共享内存。`fields=None` 存储整个序列化 TH-ACP 请求；传入 JSON Pointer 路径元组（例如 `('/patient_dialog',)`）则只存储 `payload.text` 中选定的字符串字段。低于阈值的载荷自动使用原 Pipe 路径。父进程按病例和内容摘要去重；Pipe 只携带引用、大小、摘要、病例及消息编号等元数据。子进程校验引用和内容后恢复原报文，再调用原 Handler。TH-ACP 的消息 ID、关联 ID、trace ID 和业务结果保持不变；Doctor→Expert 的第六阶段二进制传递也继续有效。

完整病例可用 `python3 main.py --runtime process --shared-dialog --stage test --count 10` 验证。此模式把 Patient 对话在送往 Expert、Retrieval、Doctor 的三次请求中复用同一共享段；真实药物标签只送往 Expert，不写入共享段。默认不带 `--shared-dialog` 时仍全部走既有 Pipe 路径。共享段由主进程持有，在病例结束、中止、通信超时、子进程退出或会话关闭时解除映射和删除；子进程只在处理请求期间附加和读取。损坏引用返回关联的 `ERROR`，连接故障沿用第五阶段的会话失败处理。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_shared_transport -v
PYTHONDONTWRITEBYTECODE=1 python3 measure_shared_transport.py
```

探索性统计写入 `runs/stage7_shared_transport.json`，包含实际 Pipe 帧字节、每例共享内存峰值、共享段写入和读取字节、复制时间，以及固定模型病例和无业务处理 Echo 的往返时间。病例往返时间包含 Agent 工作；复制计时只覆盖共享段读写，不覆盖 JSON 编解码或 Pipe 内核复制。当前共享传输由主进程选择并用于 Runtime→Agent 请求，Agent→Runtime 回复仍使用 Pipe；这次未改造其他业务链路或最终协议编码。

## 第八阶段第一步：Retrieval 历史回复增量模式

进程模式可添加 `--incremental-history`。默认仍是完整回复：`retrieval.history` 返回 `case_id`、`dataset_path`、`patient_dialog`、`count` 和新检索的 `history`。增量模式只在该回复上发送 `case_id` 与 `history`；主进程保留先前 Patient 回复中经过校验的无标签上下文，核对增量回复的病例编号和字段后，恢复与完整模式逐字相同的 Doctor 输入。病例完成或失败时清除临时上下文。其他报文、Agent 数量和流程不变；`--shared-dialog` 可独立选择。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_incremental_history -v
PYTHONDONTWRITEBYTECODE=1 python3 measure_incremental_history.py
python3 main.py --runtime process --incremental-history --stage test --count 10
```

探索性统计写入 `runs/stage8_incremental_history.json`。两种模式均使用普通 Pipe、同一批确定性模型病例，在独立测量进程中重置报文编号；报告含实际 Pipe 帧字节、Retrieval 往返时间、子进程回复 JSON 编码、父进程 Trace 再序列化及增量恢复时间。往返时间包含历史检索，不能单独归因于传输。测试模式冻结历史记忆，不写入经验文件。

## 第八阶段第三步：Retrieval 药物说明回复增量模式

进程模式可添加 `--incremental-medicines`，默认关闭。完整模式的 `retrieval.medicines` 回复包含 `case_id`、`recommended_drugs`、`actual_drugs` 和 `medicine_introductions`；增量模式只在这一条回复中发送 `case_id` 与新增的 `medicine_introductions`。主进程从本例 `expert.evaluate` 的已校验回复保存两组药名，核对 Retrieval 回复的病例编号、trace ID、请求关联 ID 和药物说明键集合，再还原与原模式逐字相同的 `expert.review` 输入。暂存药名仅在 Expert 评判后建立，病例结束或异常时清除。`--incremental-history` 和 `--shared-dialog` 可独立组合；其余通信链路不变。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_incremental_medicines -v
PYTHONDONTWRITEBYTECODE=1 python3 measure_incremental_medicines.py
python3 main.py --runtime process --incremental-medicines --stage train --count 10
```

探索性对照写入 `runs/stage8_incremental_medicines.json`。两组测量均使用普通 Pipe 和确定性模型，在独立进程中使用相同的报文编号；报告分别列出实际 Pipe 帧字节、Retrieval 往返、子进程回复 JSON 编码、父进程 Trace 编码以及增量模式的上下文解析与还原开销。药物说明正文仍占主要字节，因此只去掉重复药名列表的收益有限。

## Retrieval → Expert 药物说明共享内存

进程模式可选 `--shared-medicines`，默认关闭，可与 `--shared-dialog`、`--incremental-history`、`--incremental-medicines` 同时使用。Retrieval 查询药物说明后把 `medicine_introductions` 的 JSON 字节写入病例级共享段，Pipe 回复仅携带引用和病例字段；主进程核对 TH-ACP 消息关联及引用，并把引用转给 Expert。Expert 校验共享段内容的 SHA-256 后恢复原始 `expert.review` 输入。共享段在病例结束、中止或会话关闭时释放；进程失效后由主进程回收已登记的引用。Doctor 的输入不变。

```bash
python3 main.py --runtime process --shared-dialog --incremental-history --incremental-medicines --shared-medicines --stage train --count 10
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_shared_medicines -v
cd /data/liust/caozuoxitong/test && PYTHONDONTWRITEBYTECODE=1 python3 run_shared_medicines_ab.py
```

独立 A/B 结果保存在 `/data/liust/caozuoxitong/test/results/shared_medicines_ab/`。通信 Token 同时计入 Pipe 文本和 Expert 从共享段读取的药物说明文本；共享内存读写字节另列，不视为零成本。

```bash
cd /data/liust/caozuoxitong/tiny_hospital
python3 main.py --count 3 --stage train
python3 main.py --count 3 --stage test
```

训练阶段默认按 `data/traincut.txt` 的原始顺序运行，测试阶段默认使用 `data/testcut.txt`，并自动从 `data/traincut.txt` 恢复历史经验的来源对话。`--dataset` 仍可显式覆盖当前运行数据集；非默认测试集不会自动推断它对应的训练集。实际模型需配置 `DLUT_API_KEY` 或 `tool/llm_client.py` 的客户端设置。本地模拟测试无需网络：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

这是教学模拟，推荐结果不用于实际用药。

## TH-ACP 原生结构化业务载荷

四进程模式可选 `--structured-data`，默认关闭。开启后，病例业务 JSON 对象在 Pipe 上使用 `payload.data`，普通指令及确认文本仍用 `payload.text`；`retrieval.retrieve` 的 `parameters` 特例不变。现有 Agent Handler 仍接收原先的 JSON 字符串，传输边界负责对象与字符串转换。二进制推荐侧载荷、共享对话与药物说明引用、增量回复都可组合使用。

```bash
python3 main.py --runtime process --structured-data --shared-dialog --shared-medicines --incremental-history --incremental-medicines --stage test --count 3
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_structured_payload -v
cd /data/liust/caozuoxitong/test && PYTHONDONTWRITEBYTECODE=1 python3 run_thacp_data_ab.py
```

三例确定性 A/B 结果单独保存在 `/data/liust/caozuoxitong/test/results/thacp_data_ab/`。协议结构和原有报文路径详见 [TH_ACP_PROTOCOL_ANALYSIS.md](TH_ACP_PROTOCOL_ANALYSIS.md)。

## 赛题统一对比评测

`benchmark_competition.py` 为赛题提供一个离线、可复现的统一入口。它在同一批
10 个固定病例和相同确定性角色策略下比较：

- `text_baseline.py` 的纯自然语言协作（每次交接重复角色、任务与完整上下文）；
- 本项目实际 TH-ACP 结构化业务流程；
- 四 Agent 进程 Runtime 中 Doctor → Expert 的二进制决策状态传递。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 benchmark_competition.py
```

命令会生成 `runs/competition_report.json` 和 `runs/competition_report.md`，包含相同
病例和相同推荐结果的校验、消息数、通信字符/Token 估算、协议完整报文字节、每例本机
时间，以及二进制状态交接帧的实际字节数。若只需快速运行文本与协议对比，可加
`--skip-process` 跳过多进程二进制测量。这里的 Token 是稳定的通信估算，模型桩不产生
真实 API Token；报告明确不将本机耗时外推为真实模型服务时延。
