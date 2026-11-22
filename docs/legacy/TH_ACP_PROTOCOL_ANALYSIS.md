> 历史文档：保留设计演进资料，命令、路径、模型及行为可能已过时。当前运行以根目录 README.md 和源码为准。

# Tiny Hospital TH-ACP `payload.text` 基线协议分析

> 本文记录原 `payload.text` 实现的分析快照。后续新增的可选 `payload.data` 模式见 [README.md](README.md) 和 [三例 A/B 报告](../test/results/thacp_data_ab/report.md)；本文的“当前”均指该改造之前的基线代码与日志。

> 范围：截至 2026-10-10 的代码和已有日志；只读分析，没有启动模型或重跑实验。这里区分 **TH-ACP 逻辑报文**、**Pipe 物理帧**、**共享内存内容**。`runs/` 当前为空，现存实验日志只记录长度、摘要和统计，不保存完整业务报文。下文完整 JSON 示例由当前 `AgentPacket.to_json()` 依据现有字段及第 1 例结果重建，`示例对话`、`示例判断依据`、`示例说明` 为演示文本，**不是原始网关抓包**。日志实测数字另列。

## 1. 报文结构和校验

所有 TH-ACP JSON 报文都是严格的两个顶层对象：

```json
{"head":{"correlation_id":null,"message_id":1,"message_type":"REQUEST","source":"System","trace_id":1,"version":"TH-ACP/1.0"},"payload":{"capability":"patient.request_history","text":"请提供对话和目标药物数量。"}}
```

`AgentPacket.to_json()` 使用 UTF-8 可表示的中文、紧凑分隔符和 `sort_keys=True`；实际字节由调用方再 `.encode("utf-8")`。`ProtocolParser` 要求顶层恰好只有 `head`、`payload`，`head` 恰好包含以下六项。代码：[protocol.py](protocol.py) `PacketHead`、`AgentPacket.to_json`、`ProtocolParser._validate`。

| `head` 字段 | 当前含义和约束 |
| --- | --- |
| `version` | 固定 `TH-ACP/1.0`。 |
| `message_id` | 当前 trace 内正整数；主进程生成业务请求及预留子进程回复 ID。不同 trace 可重复。 |
| `trace_id` | 病例标识；`HELLO/HELLO_ACK` 固定 0，`REQUEST/RESPONSE` 必须非 0；`ERROR` 可为 0。不是业务 `case_id`。 |
| `message_type` | `HELLO`、`HELLO_ACK`、`REQUEST`、`RESPONSE`、`ERROR` 之一。 |
| `source` | 非空字符串；原始业务发送方或产生回复的 Agent；Runtime 错误及握手确认用 `Runtime`。 |
| `correlation_id` | 正整数或 `null`；`RESPONSE/HELLO_ACK` 必填，指向被回复报文的 `message_id`。业务链后续请求/转发也常指向上一步报文。 |

| 类型 | `payload` 的精确键集合 | 额外规则 / 实际用途 |
| --- | --- | --- |
| `REQUEST` | `{"capability": str, "text": str}` | 四进程病例使用的形式；`text` 可以是普通指令或序列化业务 JSON。 |
| `REQUEST` 特例 | `{"capability":"retrieval.retrieve", "parameters": object}` | 同步 Runtime 中结构化检索接口；`parameters` 要有 `candidate_diseases`（1–3 个不重复名称）、`knowledge_type:"symptoms"`、`include_history:bool`，可有 `memory_types`。当前 `ProcessHospitalSession` 不走此特例。 |
| `RESPONSE` | `{"capability": str, "text": str}` | Handler 输出也是字符串；四进程业务回复由子进程构造，后续编排步骤可能再生成一个新的 `REQUEST` 或 `RESPONSE`。 |
| `ERROR` | `{"error": 非空 str}` | Handler 异常由子进程返回关联 ERROR；超时、连接断开及无效路由由 Runtime 创建 ERROR。 |
| `HELLO` | `{"capabilities": [str, ...]}` | 非空、无重复；Agent 注册声明，`trace_id=0`。 |
| `HELLO_ACK` | `{"status":"ok", "capabilities":[str, ...]}` | Runtime 接受注册，`correlation_id` 指向 HELLO，`trace_id=0`。 |

`HELLO` 示例：`{"head":{"correlation_id":null,"message_id":1,"message_type":"HELLO","source":"DoctorAgent","trace_id":0,"version":"TH-ACP/1.0"},"payload":{"capabilities":["doctor.recommend"]}}`。对应 `HELLO_ACK` 把 `source` 改为 `Runtime`、`message_id` 设为下一注册 ID、`correlation_id` 设为 1、`message_type` 设为 `HELLO_ACK`，`payload` 为 `{"status":"ok","capabilities":["doctor.recommend"]}`。在四进程实现中，这两个报文由主进程的 `register_agent()` 内部处理和记录；子进程真正的就绪信号是 Pipe 帧 `READY`，**不是 HELLO_ACK**。`control_agent()` 的 `begin_case/snapshot/finish_case/abort_case` 以及 `STOP` 也不是 TH-ACP 报文。

`ERROR` 格式示例（Handler 抛错时）：`{"head":{"correlation_id":11,"message_id":12,"message_type":"ERROR","source":"ExpertAgent","trace_id":1,"version":"TH-ACP/1.0"},"payload":{"error":"ValueError: 示例错误"}}`。这只是代码路径产生的格式示意，不是发生过该错误的日志。父进程校验回复类型、预留 `message_id`、`trace_id`、`correlation_id`、`source` 和 capability；关联失败会使 Runtime 进入 `FAILED` 并回收 worker。详见 [ipc_runtime.py](ipc_runtime.py) `_managed_agent_worker`、`MultiprocessRuntime._dispatch`、`_fail_transport`。

## 2. 四条重点链路的完整报文示例

下面八条是**按当前代码重建的完整 TH-ACP JSON 字符串**；第 1 例已保存的最终结果包含推荐和真实药物 `酮康唑`，但未保存对应的完整对话、推理文本、说明书及帧字节，因此这些正文用短示例替代。编号和关联链沿用现存第 1 例 `communication.jsonl` 中 `9→16` 的顺序。这里展示的是每次业务投递的请求和子进程回复：`doctor.recommend` 的输入在当前流程中是 `RESPONSE`，`expert.review` 的输入也为 `RESPONSE`，不能按字面 capability 名统一称为 `REQUEST`。真实日志可查 [communication.jsonl](../test/results/q10_gateway_pair/logs/all_features/communication.jsonl)。

### `doctor.recommend`：Retrieval → Doctor → Runtime

```json
{"head":{"correlation_id":8,"message_id":9,"message_type":"RESPONSE","source":"RetrievalAgent","trace_id":1,"version":"TH-ACP/1.0"},"payload":{"capability":"doctor.recommend","text":"{\"case_id\": 1, \"dataset_path\": \"/data/liust/caozuoxitong/tiny_hospital/data/test.txt\", \"patient_dialog\": \"示例对话\", \"count\": 1, \"history\": {\"golden_experiences\": [], \"reflections\": []}}"}}
{"head":{"correlation_id":9,"message_id":10,"message_type":"RESPONSE","source":"DoctorAgent","trace_id":1,"version":"TH-ACP/1.0"},"payload":{"capability":"doctor.recommend","text":"{\"case_id\": 1, \"reasoning_summary\": \"示例判断依据\"}"}}
```

共享对话开启且文本达到阈值时，第一条在 Pipe 上变成“轻量 TH-ACP 报文 + shared-fields 封套”；Doctor 子进程恢复原字段后才调用原 Handler。第二条是**仅文本部分**；药物推荐另有原始二进制侧载荷，见第 4 节。关闭 `binary_state` 时，Doctor 回复的 `text` 则包含 `case_id`、`recommended_drugs`、`reasoning_summary` 三项。

### `expert.evaluate`：Doctor → Expert → Runtime

```json
{"head":{"correlation_id":10,"message_id":11,"message_type":"REQUEST","source":"DoctorAgent","trace_id":1,"version":"TH-ACP/1.0"},"payload":{"capability":"expert.evaluate","text":"{\"case_id\": 1, \"reasoning_summary\": \"示例判断依据\"}"}}
{"head":{"correlation_id":11,"message_id":12,"message_type":"RESPONSE","source":"ExpertAgent","trace_id":1,"version":"TH-ACP/1.0"},"payload":{"capability":"expert.evaluate","text":"{\"case_id\": 1, \"recommended_drugs\": [\"酮康唑\"], \"actual_drugs\": [\"酮康唑\"]}"}}
```

第一条 Pipe 帧也有相同二进制推荐状态。Expert 解码后在本地恢复 `recommended_drugs`，然后调用原 `evaluate()`；真实标签保存在 Expert 私有病例状态中，Doctor 输入没有 `actual_drugs`。`expert.evaluate` 回复之后才同时出现推荐药和真实药名。

### `retrieval.medicines`：Expert → Retrieval → Runtime

```json
{"head":{"correlation_id":12,"message_id":13,"message_type":"REQUEST","source":"ExpertAgent","trace_id":1,"version":"TH-ACP/1.0"},"payload":{"capability":"retrieval.medicines","text":"{\"case_id\": 1, \"recommended_drugs\": [\"酮康唑\"], \"actual_drugs\": [\"酮康唑\"]}"}}
{"head":{"correlation_id":13,"message_id":14,"message_type":"RESPONSE","source":"RetrievalAgent","trace_id":1,"version":"TH-ACP/1.0"},"payload":{"capability":"retrieval.medicines","text":"{\"case_id\": 1, \"medicine_introductions\": {\"酮康唑\": {\"说明\": \"示例说明\"}}"}}
```

上面回复对应 `incremental_medicines=True`、`shared_medicines=False`。若两者都关闭，回复正文还重复带 `recommended_drugs`、`actual_drugs`。若开启 `shared_medicines`，回复正文中的 `medicine_introductions` 改为 `null`，另有 `shared_medicine_ref`；见第 5 节。Retrieval 只在输入来源为 `ExpertAgent` 时接受此请求。

### `expert.review`：Retrieval → Expert → Runtime

```json
{"head":{"correlation_id":14,"message_id":15,"message_type":"RESPONSE","source":"RetrievalAgent","trace_id":1,"version":"TH-ACP/1.0"},"payload":{"capability":"expert.review","text":"{\"case_id\": 1, \"recommended_drugs\": [\"酮康唑\"], \"actual_drugs\": [\"酮康唑\"], \"medicine_introductions\": {\"酮康唑\": {\"说明\": \"示例说明\"}}"}}
{"head":{"correlation_id":15,"message_id":16,"message_type":"RESPONSE","source":"ExpertAgent","trace_id":1,"version":"TH-ACP/1.0"},"payload":{"capability":"expert.review","text":"{\"case_id\": 1, \"recommended_drugs\": [\"酮康唑\"], \"actual_drugs\": [\"酮康唑\"], \"exact_match\": true, \"reflection\": null, \"golden_experience\": null}"}}
```

`incremental_medicines=True` 时，主进程使用已校验的本病例 `recommended_drugs`、`actual_drugs` 与 Retrieval 增量回复组合后才发送第一条。共享药物说明开启时，第一条正文中的说明映射也改为 `null` 加引用，Expert 在本地读出并恢复原始业务输入。上面回复的字段集合与现存第 1 例结果相同；实际长文本仍以保存的结果为准。

## 3. 业务正文编码与解码次数

正常文本链路遵循：Agent 方法构造业务对象 → `json.dumps(..., ensure_ascii=False)` 得到 **业务 JSON 字符串** → 放进 `payload.text` → `AgentPacket.to_json()` 再对整个报文编码。于是外层 `text` 中的内层引号被转义；解析时先由 `ProtocolParser.parse()` 对外层 JSON 执行一次 `json.loads()`，Handler 再对 `text` 执行一次 `json.loads()`。从一个业务对象到另一个 Agent 的对象，典型路径是 **两次 JSON 序列化、两次 JSON 反序列化**，不包括 Trace 为统计而再次调用 `to_json()`、增量恢复、共享封套处理等附加工作。普通指令文本无内层业务 JSON；`retrieval.retrieve` 特例的 `parameters` 也没有 `payload.text` 二次编码。

四进程路径还在父进程 `Runtime.deliver()` 中先序列化并解析一次用于校验和 Trace，然后 `_dispatch()` 再 `packet.to_json().encode()` 放上 Pipe，子进程再次解析。这意味着**实际函数调用次数高于上述抽象的两层编码**。共享对话字段模式额外解析正文、写出轻量正文，并在子进程读回后再序列化完整正文；增量历史和药物回复还在主进程解析、校验、拼回业务 JSON。具体调用见 [protocol.py](protocol.py) `Runtime.deliver`、[ipc_runtime.py](ipc_runtime.py) `_dispatch`、[shared_transport.py](shared_transport.py) `prepare_shared_request/resolve_shared_request`、[hospital_process.py](hospital_process.py) `_restore_history_response/_restore_medicine_response`。

## 4. 二进制药物决策状态如何上 Pipe

当前四进程 `binary_state=True`：Doctor Handler 原样生成 `{"case_id","recommended_drugs","reasoning_summary"}`，`HospitalAgentEndpoint.handle()` 用 [decision_vector.py](decision_vector.py) `DecisionVectorCodec.encode()` 将药名按 `medicine/medicine.json` 固定编号编码。`payload.text` 只留 `case_id` 和 `reasoning_summary`。同一原始 `bytes` 先作为 `doctor.recommend` **回复**侧载荷返回父进程，再作为 `expert.evaluate` **请求**侧载荷发给 Expert；两次物理传递、一次逻辑 Doctor→Expert 交接。Expert 解码、验证目录指纹与位图/顺序，再填回药名数组。

每个二进制块为 `!4sBHB8s` 头（16 字节：魔数 `THDV`、版本 1、目录药物数、所选数量、8 字节目录指纹）+ `ceil(目录数/8)` 字节位图 + 每种推荐药 1 字节编号顺序。当前目录有 70 种药，位图 9 字节；第 1 例一药为 **26 字节**。顺序列表保证还原原始推荐顺序。它既不在 `payload.text` 中，也不是 JSON 数组/Base64。

二进制 Pipe 负载：`\x00THDV`（5 字节传输标记）+ `!I`（4 字节 TH-ACP JSON 长度）+ 完整 TH-ACP JSON bytes + 决策 `bytes`。请求前面还有主进程预留的 `!Q response_id`（8 字节）；每个 Pipe 帧另有 `!i` 长度头（4 字节）。因此二进制请求除 TH-ACP 和决策块之外固定 **4+8+5+4=21 字节**；二进制回复为 **4+5+4=13 字节**。这些是应用层帧的长度，不含操作系统内部开销。父进程记录在 `binary_frame_stats` 中，异常或关联不符会失败并清理会话。

## 5. 两种共享内存路径

### `shared-dialog`：通用字段共享封套

`ProcessHospitalSession` 在三个请求上指定 `shared_fields=("/patient_dialog",)`：`expert.receive_case_truth`、`retrieval.history`、`doctor.recommend`。父进程 `MultiprocessRuntime.deliver_shared()` 调用 `prepare_shared_request()`；达到 `shared_min_bytes`（默认 256）时，父进程 `SharedPayloadArena` 创建以 `thipc_` 开头的共享段，写入 **UTF-8 对话原文**，按 `(trace_id,SHA-256)` 去重。若低于阈值，则完整 TH-ACP 报文走普通 Pipe。

字段模式的 Pipe 内容（示意，`REF` 为下面的引用对象）为：

```text
Pipe 4字节长度头
+ 8字节预留回复ID
+ b"\x00THSM" + 4字节元数据长度
+ {"trace_id":1,"message_id":9,"mode":"fields","fields":[{"path":"/patient_dialog","ref":REF}]}
+ 轻量 TH-ACP JSON：payload.text 内 patient_dialog=null，其余字段保留
```

`REF` 的精确字段集合是 `{"name":"thipc_<32位十六进制>","size":正整数,"sha256":"<64位十六进制>","trace_id":1}`。这里的 `REF` **不在 TH-ACP payload.text 里**，而在外层 Pipe 封套元数据里。Agent 子进程先验证引用字段、trace、段长度与 SHA-256，读回原文、替换 `null`、重新构造原始 TH-ACP 报文，再调用 Handler。回复加 `b"\x00THSR" + !QQ(读取字节数,复制纳秒)`（21 字节）作回执，父进程核对读取量。父进程在病例正常结束、失败或关闭时 `close/unlink`；共享池记录释放错误。`fields=None` 的通用接口还可把**整张 TH-ACP 报文**放共享段，对应 `mode:"packet",ref:REF`，但当前 `shared-dialog` 调用的是 `fields` 模式。[shared_transport.py](shared_transport.py) 中 `prepare_shared_request`、`resolve_shared_request`、`SharedPayloadArena.release` 定义了这套格式。

### `shared-medicines`：Retrieval 产生、父进程转发引用

此开关默认关闭。开启时，主进程在病例开始为 Retrieval **预留段名**，通过非 TH-ACP `begin_case` 管理帧交给 Retrieval。Retrieval 的 `retrieve_medicines()` 仍产生原说明映射；随后 Retrieval 子进程将 `json.dumps(medicine_introductions, ensure_ascii=False).encode("utf-8")` 写入共享段。它的 TH-ACP 回复 `payload.text` 解析后形如：

```json
{"case_id":1,"medicine_introductions":null,"shared_medicine_ref":{"name":"thipc_<32位十六进制>","size":1012,"sha256":"<64位十六进制>","trace_id":1}}
```

上面 `size:1012` 来自已保存的第 1 例 `shared_write_bytes`；真实段名和摘要未被现存日志保存，尖括号是占位符，**不是可直接解析的实际引用**。是否额外有 `recommended_drugs/actual_drugs` 取决于 `incremental_medicines`；全功能模式中没有这两项。Runtime 验证回复结构、预留段名、段大小并登记段；主进程核对消息关联和病例上下文、补回两组药名，把同一个引用放入 `expert.review` 的 `payload.text`，仍用普通 TH-ACP Pipe 帧传过去。Expert 再核验大小和 SHA-256，读共享 JSON，填回说明映射后调用原 `review()`。正常病例结束、失败、会话关闭都释放；若 Retrieval 创建段后尚未回复即崩溃，主进程尝试用预留段名附着并回收。原生产者和主进程都执行容忍 `FileNotFoundError` 的清理路径。这个引用是**业务正文内的字段**，不同于上面外层 `THSM` 共享封套。[hospital_process.py](hospital_process.py) `HospitalAgentEndpoint.handle`、`ProcessHospitalSession.run_case` 和 [ipc_runtime.py](ipc_runtime.py) `_register_forwarded_medicines/_check_forwarded_medicines/_release_forwarded` 负责此链路。

## 6. 已有日志支持的开销边界

来源是已有真实网关三例、全功能开启的 [communication.jsonl](../test/results/q10_gateway_pair/logs/all_features/communication.jsonl) 与 [all_features.json](../test/results/q10_gateway_pair/all_features.json)；未重跑。`Trace.packet_bytes` 是 `AgentPacket.to_json()` 的**逻辑报文** UTF-8 长度，不能当实际 Pipe 帧长度：共享字段的 Trace 仍计入恢复前的完整报文，二进制和外层引用又不在 TH-ACP JSON 内。实际 Pipe 字节以实验捕获的 `request_pipe_bytes/response_pipe_bytes` 为准，含每帧 4 字节长度头。

| 范围 / 口径 | 现存实测值 | 能说明什么 |
| --- | ---: | --- |
| 四条重点链路，病例 1，Trace 逻辑报文 `packet_bytes` / 原始 `message_bytes` | `doctor.recommend` 892+511 / 652+321；`expert.evaluate` 509+275 / 321+81；`retrieval.medicines` 278+437 / 81+228；`expert.review` 506+347 / 295+149 | 单条完整报文大小和业务 `text` 原始 UTF-8 长度。 |
| 四条重点链路，病例 1，实际 Pipe 请求+回复 | `doctor.recommend` 630+571；`expert.evaluate` 556+279；`retrieval.medicines` 290+441；`expert.review` 518+351 字节 | 含 Pipe 帧头、回复 ID、二进制封套/载荷；共享内容只以引用形式出现在 Pipe 上。 |
| 全部 3 例、六条 Agent→Agent 链路、36 个请求/回复，实际 Pipe | **18,068 字节** | 不含 System→Patient、End 和管理帧。 |
| 同范围二进制决策块 | **168 字节**、6 次物理帧附带、3 次逻辑传递 | 已包含在上行 Pipe 字节，不可再相加。 |
| 同范围共享内存写入 / 读取 | **13,477 / 18,083 字节** | 含共享对话及共享药物说明；是额外通信和复制成本，不可当作数据消失。 |
| 同范围 `shared_reference_bytes` | **6,909 字节** | 实验字段实际表示*被引用的数据量*，不是 Pipe 中引用元数据本身的字节长度，不能作为“引用开销”解读。 |
| 全部 3 例 Trace 病例报文（含 System、End），逻辑报文 / 原始业务正文 | **31,872 / 20,842 字节**，差 **11,030 字节** | 差值混合协议头、payload 键、capability、JSON 引号与转义增长；**不是纯协议头字节**，也不是物理 Pipe 开销。 |

固定可精确拆出的**传输常量**：每 Pipe 帧 4 字节长度头；业务请求另有 8 字节预留回复 ID；二进制侧载荷封套 9 字节；`THSM` 引用封套固定前缀 9 字节，另加可变 JSON 元数据和可能的轻量 TH-ACP 报文；`THSR` 回执固定 21 字节。`shared_medicine_ref` 位于嵌套 JSON 字符串，具体字节数受 JSON 空格、转义、`trace_id` 位数和字段内容影响。现存日志只保留各报文长度及部分摘要，不保留原始 TH-ACP/封套 bytes，所以**无法精确反推出协议头、payload 封装、共享引用元数据各自所占的实测字节数**。不能将 `packet_bytes-message_bytes` 伪装成纯协议头；也不能把协议模板 Token 当作字节数。模型 API Token 与这里的通信字节无关。实验汇总见 [summary.json](../test/results/q10_gateway_pair/summary.json)。

## 7. 代码位置与调用关系

```text
ProcessHospitalSession.run_case  [hospital_process.py]
  ├─ control_agent(begin_case / finish_case / abort_case)  [ipc_runtime.py]  # 管理帧
  └─ _deliver → AgentPacket.create → Runtime.deliver  [protocol.py]
       └─ MultiprocessRuntime._dispatch  [ipc_runtime.py]
            ├─ prepare_shared_request / _pack_binary_message（按选项）
            ├─ _send_frame → 子进程 _managed_agent_worker
            │    ├─ resolve_shared_request / _unpack_binary_message
            │    ├─ ProtocolParser.parse → HospitalAgentEndpoint.handle
            │    │    ├─ Patient/ Retrieval/ Doctor/ Expert 原业务 Handler  [agent/*.py]
            │    │    ├─ DecisionVectorCodec  [decision_vector.py]
            │    │    └─ SharedPayloadArena / read_reference  [shared_transport.py]
            │    └─ AgentPacket(RESPONSE 或 ERROR) → _send_frame
            ├─ _recv_frame → 关联校验 / ERROR 返回 / 失败清理
            └─ Trace.packet / append_communication_records  [tool/tracing.py]
```

同步 `Runtime.deliver` 仍负责本地 Handler 调用；四进程的 `MultiprocessRuntime` 仅对注册的进程 capability 覆盖 `_dispatch`，`case.complete` 仍为父进程本地 `End`。`ProcessHospitalSession` 保留原业务顺序，并在 `retrieval.history` 与 `retrieval.medicines` 增量回复后于主进程恢复完整输入。上述路径没有修改记忆检索、提示词或医学逻辑。

## 8. 可观察的潜在冗余（本次未改）

1. 普通业务对象先变 JSON 字符串再嵌入外层 JSON，`text` 内引号和反斜杠重复编码；父进程校验、Pipe 写入和 Trace 统计也重复序列化同一逻辑报文。
2. `capability`、`trace_id`、消息关联存在于报文和路由/共享引用元数据；共享对话封套重复携带 `trace_id/message_id` 用于校验，属于有意的完整性开销。
3. `patient_dialog` 在病例真值、历史请求、Doctor 输入中重复出现；共享对话减少 Pipe 复制但仍产生每个接收方的共享读取。
4. 未开启增量时 `retrieval.history` 回复重复 `patient_dialog/count/dataset_path`；未开启药物增量时 `retrieval.medicines` 回复重复两组药名。
5. 二进制块从 Doctor 到父进程再到 Expert 两次通过 Pipe；编码后同时保存位图和编号顺序，顺序对业务还原有用，位图与顺序一致性用于校验。
6. `shared-medicines` 在 `retrieval.medicines` 回复及 `expert.review` 请求中重复传同一个引用；主进程需要检查和转发，现有日志无法单独精确量化这两次引用的字节数。
