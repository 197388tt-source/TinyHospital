# Patient → Retrieval 查询向量传输

正式会话在 embedding 检索下默认启用 `shared_memory` 查询传输。Patient 仅编码对话，不编码病例真实标签；Retrieval 接收向量直接检索，不再重复编码查询。Doctor/Expert 继续接收文字证据和精确药物数据。

## 运行

配置原有聊天凭据和 `EMBEDDING_API_KEY` 后，在 tiny_hospital 目录执行：

```bash
python main.py --runtime process --stage train --count 3 \
  --retrieval-mode embedding --vector-transport shared_memory
```

三种模式：

- `text`：Patient 只发送完整对话；Retrieval 在有可检索记忆时编码查询。
- `binary`：Patient 不发送对话，只将归一化向量编码为 little-endian float32，通过二进制 sidecar 返回父进程，再由父进程转发给 Retrieval。两次物理 Pipe 传递，仅支持 process Runtime。
- `shared_memory`：Patient 不发送对话，只将向量写入共享段，并经 Pipe 传递描述和引用。同步 Runtime 也支持此模式，但没有跨进程 Pipe 成本。

非 embedding 检索自动使用 text。向量模式即使经验库为空，也编码并传递查询向量，以保持固定协议链路；因此需要有效的 embedding 配置，会产生编码调用。当前病例对话由同步编排器或进程父进程保留，只在 `Retrieval → Doctor` 前与检索到的历史经验合并；不会传入 Retrieval。这样 text、binary 与 shared_memory 是互斥的查询状态传输模式。

`binary` 与 `--shared-dialog` 同时使用时，仅 Patient → Retrieval 这一跳不共享对话（现有帧不同时支持两种侧载荷）；其他共享对话链路照常。`shared_memory` 支持增量回复、共享药物和结构化 data。

## 数据契约与清理

业务对象新增 `query_embedding` 描述，版本为 1，包含 kind、storage、模型预处理指纹、dimension、dtype、normalized、字节长度、内容 SHA-256、规范化对话 SHA-256 和 trace_id。共享模式还包含 reference，其 name、size、sha256、trace_id 复用现有共享内存引用格式。

接收方校验模型、病例归属、字节长度、摘要、有限数值及单位范数；文本模式还校验对话内容。向量模式下 Retrieval 无权读取原始对话，因此对话哈希绑定由持有原文的编排器与 Patient 负责，Retrieval 继续校验向量字节 SHA-256 和 trace ID。Retrieval 在回复 Doctor 前移除向量描述。共享段名称由父进程在病例开始前预留，Patient 创建并写入；Patient 在病例结束/中止/进程退出时释放，父进程按预留名称兜底回收。

Patient 有最多 256 条的会话内查询缓存。每条经验记忆只对其来源病例对话编码，查询向量也只由当前对话生成；余弦相似度因此比较的是“当前对话 ↔ 历史对话”。命中后再根据记录 ID 返回对应的经验或反思文字，经验正文不参与相似度计算。因此只修改经验文字而来源对话未变时，会复用原对话向量。向量持久化在 SQLite；Expert 保存后自动同步，Retrieval 在训练模式下下次查询检查文件变化。测试阶段冻结记忆。Expert → Retrieval 的主动向量更新消息尚未实现。

## 观察指标

结束时的历史检索统计新增：

- `vector_transport`：实际查询传输模式。
- `query_sender.query_encodings/backend_calls/encoding_ms/cache_hits`：Patient 编码与缓存。
- `query_sender.shared_write_bytes`：共享向量实际写入量。
- `vector_receive.received_vectors/shared_read_bytes/binary_bytes`：Retrieval 接收量。
- `embedding.query_encodings`：Retrieval 自己编码的查询数量；向量传输模式应为 0。
- `embedding.search_backend`：当前为 `numpy_exact`，表示连续 `float32` 矩阵全量精确检索。
- `embedding.matrix_rows/matrix_dimension/matrix_bytes`：当前记忆向量矩阵的规模和数值载荷字节数。
- `embedding.searches/search_ms`：矩阵乘法的累计次数和耗时；完整 Retrieval 耗时仍看外层的 `average_latency_ms`。

4096 维向量每个 16384 字节。binary 的发送、转发两次物理字节计入 Runtime 的 binary_frame_stats 和 transport_frame_stats；query_sender.binary_bytes 只记录 Patient 生成的逻辑载荷量。共享向量字节使用上述独立计数，不混入文本 Token，原日志的正文/报文字节不代表全部实际传输量。这些统计不包含 Expert 端记忆编码的 API 用量。

## 用户自行验证

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p test_semantic_retrieval.py -v
```

该测试启动本机模拟 API，不需要本地 embedding 模型，不访问学校 API；用于协议和流程验证，不衡量真实语义效果。

真实 embedding 的三模式对照（聊天使用固定模型、经验冻结，不用于质量评估）：

```bash
cd ../test
python run_vector_transport_ab.py --count 3
```

使用非空、有合法来源的经验库才可验证语义命中；可以传 `--golden`、`--reflection` 和 `--dataset`。首次每模式独立冷缓存，再次运行会复用缓存；比较冷热结果时请使用新的 `--output`。结果在 results/vector_transport；统计分别列出 Pipe 字节与共享向量读写，不将单次耗时认定为性能收益。

本次改动仅完成静态检查；按用户要求未执行测试或真实模型调用。

## 检索相关性与专家诊断

embedding 与 Jaccard 分开设置相似度阈值：前者默认 0.60，后者保持 0.15。0.60 是待校准的保守初值，不代表已验证的准确率；可能减少误召回，也可能漏掉相关经验。可通过 `--embedding-min-similarity 0.60` 调整。低于阈值时返回空历史，不用无关结果凑满 Top-K。

每次查询记录全部候选的来源病例、种类、相似度、阈值及处理原因。控制台最多显示分数最高的5项，完整数据默认追加到 `runs/diagnostics.jsonl`（随 llm_usage 文件目录变化）。原因包括 selected、below_threshold、current_case、identical_dialog、duplicate_dialog、outside_top_k、disabled、limit_zero。日志中的 hit 仅表示返回了内容，不能作为语义正确率。

专家调用记录 model_call_started / model_call_finished、尝试编号与 elapsed_ms。专家评审记录 accepted、unsupported（模型明确表示证据不足）、validation_rejected、api_error、timeout；memory_sync 单独记录保存/向量同步耗时及结果。API 失败时 Token 标为未知，不复用前次调用的 usage；llm_usage 的 case_total 会给出 failed_calls 和 usage_complete。若子进程被父进程超时终止，可能只有 started 事件；应结合父进程超时报错排查。

诊断日志不保存提示词和完整模型回复。accepted 仅表示通过现有程序校验，不代表经验已被人工确认。主流程阈值改变后，历史结果不能与之前直接当作同条件实验对比。

## 超时与 Doctor 日志

正式入口默认 `--chat-timeout 180 --agent-timeout 720`。前者设置每次聊天 API 的网络超时，后者是 process Runtime 一次 Agent 任务（可能包含最多3次内容重试及记忆编码）的总等待上限。embedding 请求仍由 `--embedding-timeout` 单独控制。网络 socket 超时不是严格的整个 HTTP 请求总时限，父进程任务上限负责最终兜底。

Doctor 也会输出模型调用开始、结束及无效输出重试日志，并写入 diagnostics.jsonl。如果 API 超时，模型诊断记录 timeout；如果任务整体超过上限，则父进程报告 capability 和配置上限。API 超时当前不自动重试；失败病例仍中止当前运行，避免悄悄丢病例或改变实验样本。调大超时只提供更多等待时间，不保证网关能正常返回。

```bash
python main.py --runtime process --stage train --count 3 \
  --retrieval-mode embedding --vector-transport shared_memory \
  --embedding-min-similarity 0.60 --chat-timeout 180 --agent-timeout 720
```
