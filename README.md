> 完整工程上传尚在进行：代码、文档及汇总结果已提供，逐病例资源等待单独公开确认。请在完整上传完成后运行数据准备与复现命令。

# Tiny Hospital：多 Agent 药物推荐与经验检索实验

本项目输入药名被 `[MASK]` 遮盖的医患对话，在药物白名单中推荐药物，并利用历史病例提炼的成功经验和错误反思辅助后续推荐。这是一个 **Python 命令行研究工程**，仓库提供代码、输入数据、最终经验和已有实验结果。

项目包含四个角色明确的 Agent。**`train` 表示生成经验库，不包含模型权重训练、LoRA 或 embedding 微调。** 医疗经验是模型生成的研究数据，没有经过临床有效性认证。

首次使用：安装环境 → `python scripts/prepare_assets.py` → 离线测试 → 按需配置模型 → 新建实验目录运行。所有命令均在项目根目录执行。

## 快速导航

- **了解结构：** [目录和入口](#1-目录和入口)、[全部目录说明](docs/DIRECTORIES.md)。
- **查看结果：** [结果总览](docs/RESULTS.md)、[离线复算指标](docs/recomputed_metrics.json)。
- **查看最终经验：** [成功经验](memories/final/golden_experience.json)、[错误反思](memories/final/reflection_memory.json)。
- **复现：** [环境配置](#6-环境配置无需-docker无需管理员权限)、[运行命令](#7-运行和复现)。

```bash
git clone https://github.com/197388tt-source/TinyHospital.git
cd TinyHospital
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -c constraints-offline.txt
python scripts/prepare_assets.py
python -m unittest discover -s tests
# 直接复算已有结果，不调用模型
python -m experiment.test --input experiment/qwen3_betterex_full2/test/progress.json --expected-count 93
```

离线逻辑测试已通过（152 通过、1 项因缺少历史基线跳过）。真实 Qwen/API 推理需要按下文配置权重和服务，未在整理环境重新运行。

## 1. 目录和入口

| 目录 / 文件 | 内容与用途 |
| --- | --- |
| `agent/` | Patient、Retrieval、Doctor、Expert；角色基类、输出校验。Doctor 的 API 和本地 Qwen 变体共享相同训练提示词，但测试输出与失败策略不同。 |
| `embedding/` | 文本预处理、API/本地编码器、SQLite 向量缓存、精确余弦检索、查询向量二进制与共享内存传输；附独立建库和 API 连通性检查工具。 |
| `history/` | `memory.py` 是经验管理与检索策略；原有 JSON 是旧经验（19 条成功、149 条反思），**不是本次整理默认的最终快照**。 |
| `memories/final/` | 当前外层脚本所使用的经验快照，52 条成功经验 + 235 条反思；`provenance.json` 记录来源与校验值。 |
| `memories/alternatives/nested_filtered/` | 嵌套旧项目中的另一套筛选经验，55 条成功 + 493 条反思。保留为独立版本，不混入当前快照。 |
| `data/` | 三个完整 JSONL 数据文件，以及准备后恢复的胃炎 `*cut.txt`、皮炎 `*piyan.txt` 子集。扩展名是 `.txt`，内容为一行一个 JSON 对象。 |
| `medicine/` | 药物候选名称 `medicine.json`、药品说明 `medicine_introduction.json`。 |
| `tool/` | 聊天模型客户端、本地 Qwen、重试、药物知识读取、日志、通信成本、原子进度和文件锁。 |
| `experiment/` | `test.py` 汇总指标，`testval.py` 校验验证集结果，`same_case_memory.py` 做同病例经验诊断；各子目录保存已有训练/测试实验。 |
| `runs/` | 原有通信统计与阶段性能报告；属于历史结果，不能当作本次机器重新测得的数据。 |
| `archive/nested/` | 从原 `embedding/tiny_hospital(5).zip` 中保留的独有历史实验、经验和消融分析；旧代码差异保存在 patch 中，不覆盖当前代码。 |
| `docs/` | 检查报告、去重清单、目录说明、复现验证记录；`legacy/` 保留旧设计文档并标明过时。 |
| `scripts/` | `prepare_assets.py` 按索引与 SHA-256 恢复去重资源，保护已修改文件。 |
| `tests/` | 标准库 unittest 测试；模型调用主要由模拟客户端/本地 HTTP 服务替代。 |
| `.github/workflows/` | GitHub Actions 自动恢复输入、运行离线测试和复算结果，不调用真实模型服务。 |
| `main.py` | 主入口，负责训练/测试、模型、检索、传输方式、实验隔离与恢复。 |
| `hospital_process.py`、`ipc_runtime.py` | 四 Agent 独立进程执行、管道通信、状态恢复、超时及资源释放。 |
| `protocol.py`、`shared_transport.py`、`decision_vector.py` | TH-ACP 协议，共享内存载荷，以及药物决策的二进制编码。决策向量不是语义 embedding。 |
| `cases.py` | 读取数据，按物理行号建立 `case_id`；提供来自标签长度的目标药物数量。 |
| `benchmark_competition.py`、`text_baseline.py` | 离线确定性通信对照；不用于证明真实模型推荐质量。 |
| `performance_*.py`、`measure_*.py` | 同步/多进程、共享内存、增量历史、增量药物说明及决策传输的专项性能实验。 |
| `capture_gateway_run.py` | 真实网关运行和调用记录；会访问配置的模型服务并产生费用。 |
| `train_qwen.sh`、`test_qwen.sh`、`val_qwen.sh`、`test_deepseek.sh` | 可追加参数的启动脚本；已改用当前虚拟环境的 `python`。 |
| `assets_manifest.json` | 去重恢复映射：6 个子集、52 个重复文件和 53 个压缩资源的目标路径、来源和哈希。 |
| `高级操作系统赛题.docx` | 原项目的任务要求，保留作为交付背景，不参与执行。 |

更细的历史实验目录说明见 [docs/DIRECTORIES.md](docs/DIRECTORIES.md)，完整文件清单见 [docs/FILE_INVENTORY.tsv](docs/FILE_INVENTORY.tsv)。

## 2. 四个 Agent 如何协作

### 2.1 角色、可见信息和职责

| Agent | 接收 | 执行 | 输出与边界 |
| --- | --- | --- | --- |
| PatientAgent | 对话、病例 ID、真实药物标签 | 向 Expert 提供真实标签；向 Doctor/检索链路提供对话与目标药物数；向量模式下负责查询编码 | 标签名称只进入 Expert 评判通道；Doctor 仍可见标签数量 `count`，因此是已知目标数量的实验设置。Patient 本身不调用聊天模型。 |
| RetrievalAgent | 测试时接收对话或查询向量；评审时接收预测/参考药物列表 | 查询历史经验；按药名取得药物说明 | Doctor 只收到经验文字列表，不收到来源病例标签、来源原对话、相似度或病例摘要；说明书通道只允许 Expert 请求。 |
| DoctorAgent | 当前对话、药物白名单、目标药物数；测试可附经验 | 调用 API 或本地 Qwen 生成推荐并校验 | 药名必须在白名单内、数量恰好符合目标、不能重复；训练返回推荐与理由摘要。 |
| ExpertAgent | Patient 私下提供的标签、Doctor 推荐与训练理由；必要时接收说明书 | 本地比较预测与标签；训练时调用 LLM 审核、生成经验 | 测试只评估，不调用 Expert LLM、不更新经验；训练仅保存满足条件的经验。 |

这是一套固定角色和固定工作流的协作系统，没有实现通用自主任务规划器。标签隔离属于协议/应用层约束，并不是操作系统权限隔离；日志和协调器仍可能持有标签。

### 2.2 训练：先推荐，再评审，形成经验

1. Patient 将对话与真实药物发给 Expert，将对话与目标数量发给 Doctor。
2. Doctor **不接收历史经验，也不进行历史检索**，输出 `recommended_drugs` 与 `reasoning_summary`。
3. Expert 比较药物集合，计算完全匹配及相应集合指标。
4. Expert 请求 Retrieval 读取“预测药物 ∪ 参考药物”的说明书。
5. 完全匹配时，Expert 审核推荐理由并尝试生成成功经验；不匹配时，审核是否存在有依据的问题，再尝试生成反思。标签不同不自动等于临床错误。
6. 将认可的经验写入当前实验目录。病例完成后提交 `train/progress.json`；支持断点恢复。

### 2.3 测试：冻结经验，检索后推荐

1. `--memory-dir` 指定来源经验；首次运行复制到独立 `--experiment-dir`，续跑使用冻结副本。
2. Patient 将对话或向量交给 Retrieval。Retrieval 从训练来源病例中检索最多 3 条经验。
3. 协调器把当前对话、目标数量和经验交给 Doctor。非文本查询模式中 Retrieval 不接收原对话，原对话由协调器保留并补回 Doctor 上下文。
4. Expert 仅本地比较推荐和标签，不生成新经验。
5. `test/progress.json` 保存完成病例及逐例结果；运行 `experiment/test.py` 才生成汇总指标。

### 2.4 Doctor 的两个实现不能全部合并

- `agent/doctor.py`：API Doctor；训练输出 JSON 对象，测试输出药名 JSON 数组；连续三次内容校验失败会抛出错误。
- `agent/doctor_Qwen.py`：本地 Qwen Doctor；训练和测试均要求药名与理由对象，测试时向下游只转发推荐；三次内容无效后返回 `doctor_failed`，记录并继续下一病例。
- 两者相同的初始化、状态清理及训练提示词已通过继承复用；不同的测试提示词、返回格式和失败语义保留。
- 本地 Qwen 设置 `enable_thinking=False`、`do_sample=False`，使用 `local_files_only=True`；模型只加载本地权重。生成时间限制不包含模型加载，也不是严格的整次请求墙钟上限。

### 2.5 通信与执行方式

`sync` 在同一进程按步骤调用；`process` 将四个 Agent 放入独立进程。TH-ACP/1.0 包含 HELLO/HELLO_ACK、能力注册与查找、REQUEST/RESPONSE/ERROR、`trace_id`、`message_id`、关联 ID 和发送方。业务默认是 `payload.text` 内的 JSON 字符串；`--structured-data` 切换为 `payload.data` 对象。

共享对话、增量历史、增量药物说明和共享药物说明是可选传输优化，需 `--runtime process`。目前本地 Qwen Doctor 只支持 `sync`。通信 Token 是正则估算；模型 API usage / Qwen tokenizer 计数单独记录，两者不能混用。

## 3. 经验库设计与“最终版”位置

### 3.1 本次整理的默认快照

**`memories/final/golden_experience.json`：52 条成功经验。**  
**`memories/final/reflection_memory.json`：235 条错误反思。**

这两份数据逐字节来自外层 `experiment/betterex/`，该目录训练进度记录 686 个已完成病例，`test_qwen.sh` 原来引用它；`experiment/qwen3_betterex_full2/` 使用相同内容。作者已明确确认以这套 betterex 作为最终版；这项版本选择不代表已经证明它在所有历史实验中效果最好。

嵌套压缩包还有另一条实验路线：`qwen_train_deepseek_full` 为 59/500 条，筛选后的 `experience` 为 55/493 条，另有 `experience_lcx`。它们不是相同文件，已保留；筛选版单独放在 `memories/alternatives/nested_filtered/`。当前正式复现使用 betterex；如研究其他版本，应单独指定 `--memory-dir` 并建立新实验目录，不能把不同版本直接合并。

### 3.2 存储结构和更新规则

每条成功经验仅保存 `case_id` 与 `experience`；每条反思仅保存 `case_id` 与 `reflection`。对话从相应训练数据集的同一物理行号恢复，不重复存入经验 JSON。**不要重排 `traincut.txt` 或删其中某一行。**

成功经验最多 200 字符；当前生成反思校验最多 200 字符、单段文字，历史存储读取兼容最多 250 字符。成功审核包含 supported 标志、病例原文引用、药物说明引用；代码能检查字段、引用是否连续出现在来源中、长度和药名等，但不能证明医学推理正确。反思校验主要是格式和长度，医学合理性仍依赖模型评审与后续人工检查。

保存某病例成功经验时清除其旧反思；保存反思时清除其旧成功经验。`training=False` 时禁止写入。未形成可靠经验的病例可以不入库，所以 686 个训练病例不等于 686 条最终经验。完整审核回复和诊断保存在实验日志/进度中；精简 JSON 本身不保存全部证据链。

### 3.3 原文件中的真实示例

以下为工程内模型生成记录的原文展示，不是新增医疗建议。

成功经验（`case_id=10`）：

```json
{
  "case_id": 10,
  "experience": "患者餐后胃胀、恶心、稍食即胀、平卧减轻，病程近一年，曾用中药后口干停药；考虑胃炎或功能性消化不良但未胃镜确诊。可试用莫沙必利促胃排空缓解上腹胀、恶心，对本品过敏者禁用，并做胃镜明确，少食产气食物。"
}
```

错误反思（`case_id=1`）：

```json
{
  "case_id": 1,
  "reflection": "如果患者以餐后下腹不适、便意/腹泻为主且大便不规律，不应忽视腹泻倾向而直接加用莫沙必利，因其主要改善上消化道动力且可致腹泻、腹痛，可能加重症状；应先评估排便异常原因，再选择不加重腹泻的处理。"
}
```

### 3.4 检索策略

成功经验与反思一起排序，默认余弦阈值 **0.60**，**合计 top-k=3**，并非两类各 3 条。分数相同时优先成功经验，再按病例 ID 排序；相同来源对话只返回一次。同一数据集中的当前病例、完全相同的规范化对话被排除。向量模式利用 Patient 描述符的文本哈希排除相同对话，不需要向 Retrieval 传递原文。

`--retrieval-mode jaccard` 为字符特征集合 Jaccard 对照，默认阈值 0.15；`none` 不返回经验。0.60 尚不能当作已优化阈值，应在验证集上校准。自定义数据文件需特别核对经验所引用的训练来源；内置 train/test/dev 的完整、cut、piyan 配对已支持，任意外部路径不会自动推断其来源。

## 4. Embedding 设计

### 4.1 编码对象与预处理

主流程对 **历史经验所属的原始训练对话** 建索引，对当前对话构造查询向量；不编码药物标签、`disease` 或 `original_dialog`。经验正文只在命中后被取出。独立 `embedding/build_vectors.py` 允许对经验正文建库，这属于另一种通用离线能力，不是主流程匹配策略。

将对话列表按换行拼接，做 Unicode NFC、统一换行、去空行与首尾空白；默认每 256 个字符分块。分块按字符而非 token，模型前缀只影响模型输入，不计入块权重。每块向量先 L2 归一化，再按块字符数加权求和，最后再次归一化：

$$q=\mathrm{normalize}\left(\sum_i |c_i|\,\mathrm{normalize}(E(c_i))\right).$$

代码拒绝零向量、非有限值和维度变化。默认批大小 32。向量维度由实际模型返回，历史 Qwen3-Embedding-8B 实验常见 4096 维，代码没有写死为 4096。相同模型、版本标记、预处理和参数生成配置指纹，防止不兼容向量混用。

### 4.2 API、本地模型与缓存

- API：OpenAI-compatible `/embeddings`，显式传完整 endpoint；凭据由 `EMBEDDING_API_KEY` 提供，与聊天密钥独立。
- 本地：SentenceTransformer，默认 CPU、仅本地文件，不信任远程自定义代码；检查 token 长度，超限报错而不静默截断。
- SQLite：`VectorStore` 持久保存规范化文本哈希、float32 小端向量、记录与命名空间；已有文本复用向量，只编码增量。
- 主检索：启动后把活跃来源向量读入连续 NumPy `float32` 矩阵，`matrix @ query` 穷举计算余弦分数；不使用 FAISS/HNSW 近似搜索。时间与矩阵大小大致随 N×d 增长。
- Patient 和 Retrieval 的查询缓存分别最多 256 条。同步、测试冻结和模型指纹约束见 `embedding/` 源码。

本次移除了可重建的旧 SQLite 缓存。第一次真实 embedding 检索需要重新编码并可能产生 API 费用；已有经验和实验预测不会丢失。

### 4.3 三种查询传输

| `--vector-transport` | Patient → Retrieval 的内容 | 编码位置 | 限制 |
| --- | --- | --- | --- |
| `text` | 原对话文本及元数据 | Retrieval | sync/process 均可 |
| `binary` | 描述符 + Pipe 二进制 float32 向量 | Patient | 需要 process |
| `shared_memory` | 描述符 + 共享内存引用 | Patient | sync/process 均可 |

非文本模式传递维度、dtype、模型指纹、字节数、SHA-256、文本哈希和 trace 绑定。接收方检查完整性、模型一致性和单位向量；不得静默重新编码。4096 维 float32 的向量主体为 16384 字节，此外还有描述符/协议开销。共享内存在操作系统内存中，不是一个永久经验文件；病例结束负责释放，SQLite 才是持久向量缓存。

## 5. 数据去重与资源恢复

| 数据版本 | 训练 | 验证 | 测试 |
| --- | ---: | ---: | ---: |
| 完整集 | 9605 | 1192 | 1199 |
| 胃炎 cut | 686 | 81 | 93 |
| 皮炎 piyan | 2194 | 274 | 275 |

九份输入文件内部均未发现完全重复的 JSON 记录；六个子集全部能从对应完整集逐字节恢复。胃炎三划分之间没有完全相同的 `dialog` 列表，但这不等同于证明不存在语义近重复或患者级泄漏。

仓库用 `.txt.gz` 无损保留三个完整集，并记录子集恢复索引；重复实验经验文件同样只保留一份实体。执行：

```bash
python scripts/prepare_assets.py --check
python scripts/prepare_assets.py
```

脚本恢复原路径、顺序、换行和哈希，已存在且相同的文件跳过；遇到已修改内容停止，不覆盖训练产物。恢复出的运行副本是有意的隔离快照，不应手动再次全局去重。`progress.json` 中的经验快照和逐例结果也是恢复/审计证据，保留其逻辑冗余。

## 6. 环境配置：无需 Docker、无需管理员权限

### 6.1 支持条件

- 推荐 Linux/WSL2，Python 3.12 虚拟环境。源码使用 Python 3.10+ 语法；本次离线验证为 Linux + Python 3.12.14 + NumPy 2.3.5。
- 完整流程依赖 POSIX `fcntl` 锁和管道文件描述符，**原生 Windows 不作为受支持复现平台**；可用 WSL2 或现有 Linux/L40 服务器。
- 纯 API、指标计算和离线通信测试不需要 GPU；本地 Qwen 需要模型权重及 PyTorch。Qwen3-8B 的 16 位权重约 16 GB，另需 KV cache、激活与框架空间；显存需求随上下文变化，24 GB 仅作起点，L40 48 GB 通常更宽裕，本次没有 GPU 实测。
- 仓库未包含模型、驱动或 CUDA。需自行准备与 GPU 驱动匹配的 PyTorch；没有 Docker 依赖。
- 原工程没有完整环境锁文件。本次 requirements 是由实际 import 和源码版本约束整理的安装范围；GPU 组合尚未实装验证，不能声称逐位复现原服务器。

### 6.2 创建环境

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -c constraints-offline.txt
python scripts/prepare_assets.py
python -m unittest discover -s tests
```

也可以使用 `conda env create -f environment.yml` 后 `conda activate tiny-hospital`。

| 场景 | 依赖文件 | 外部资源 |
| --- | --- | --- |
| API Doctor + API embedding、离线测试、全部指标和通信测量 | `requirements.txt` | 实际推理才需要对应服务与密钥 |
| 本地 Qwen Doctor | `requirements-qwen.txt` | 本地 Qwen3-8B 权重；训练还需 Expert 聊天 API |
| 本地 embedding（可同时用本地 Qwen） | `requirements-local-embedding.txt` | SentenceTransformer 可读取的本地 embedding 模型 |

```bash
# 先按驱动安装适合的 torch 构建，再安装可选依赖
python -m pip install -r requirements-qwen.txt
# 若也需要本地 embedding
python -m pip install -r requirements-local-embedding.txt
python -m pip check
```

`json`、`sqlite3`、`multiprocessing`、`urllib`、`unittest`、`fcntl` 等属于标准库，不要安装同名第三方包。主流程不依赖 openai SDK。创建完真实 GPU 环境后可自行 `pip freeze` 记录精确版本。

### 6.3 配置密钥、模型和网络

```bash
cp .env.example .env
# 编辑 .env：填自己的密钥、模型名、模型目录
set -a
source .env
set +a
```

Python 不自动加载 `.env`；必须显式 source/export。`DLUT_API_KEY` 优先，其次 `OPENAI_API_KEY`；`CHAT_BASE_URL` 和 `CHAT_MODEL` 在进程启动时读取。embedding 的 endpoint/model 使用 CLI 参数。默认网关是原项目配置，未验证当前可用性，也不保证校外网络能够访问；替换为你获授权的接口。

源文件中原有硬编码密钥已移除。原凭据应由所有者在提供方撤销/轮换，不能继续放入源码或提交历史。

模型目录使用环境变量 `QWEN_MODEL_PATH` 或 `--doctor-model-path` 指定，应包含 tokenizer、config 和全部权重分片。只调用本地 Qwen 的测试且不用 API embedding 时可以不设置聊天密钥；训练仍要调用 Expert API。

## 7. 运行和复现

先验证 CLI：

```bash
python main.py --help
python -m experiment.test --help
python -m experiment.testval --help
python benchmark_competition.py --help
```

### 7.1 快速真实推理：API Doctor + Jaccard

```bash
python main.py --stage test --runtime sync --count 3 \
  --memory-dir memories/final --experiment-dir experiment/demo_jaccard \
  --retrieval-mode jaccard --vector-transport text
```

只需聊天 API；Jaccard 用于先验证完整链路，不代表 embedding 主实验效果。

### 7.2 使用最终快照 + API embedding

```bash
python main.py --stage test --runtime process --count 93 \
  --dataset data/testcut.txt \
  --memory-dir memories/final --experiment-dir experiment/demo_embedding \
  --retrieval-mode embedding --vector-transport shared_memory \
  --embedding-model Qwen3-Embedding-8B \
  --embedding-model-version Qwen3-Embedding-8B \
  --embedding-endpoint http://aigw.dlut.edu.cn/v1/embeddings
```

把 endpoint/model/version 改为实际服务配置。原 endpoint 使用 HTTP；若服务支持 HTTPS，应采用提供方确认的 HTTPS 地址。本地 embedding 可加 `--embedding-backend local --embedding-model /你的模型目录 --embedding-model-version 你的版本标记`；主 CLI 本地编码默认 CPU，更高级的 device 配置通过 EmbeddingConfig/离线建库接口设置。

### 7.3 本地 Qwen 训练与测试

```bash
export QWEN_MODEL_PATH=/你的模型目录/Qwen3-8B
# 先少量试运行，不覆盖冻结快照
bash train_qwen.sh --count 3
# 扩展到全部 686 例并续跑
bash train_qwen.sh --resume
# 测试默认使用 memories/final，不自动改成刚训练的新经验
bash test_qwen.sh --count 3
# 如果要测试刚训练的经验，请显式指定
bash test_qwen.sh --memory-dir experiment/new_training \
  --experiment-dir experiment/new_training_test --count 93
# 验证集
bash val_qwen.sh
```

本地 Qwen 用 `sync`；生成上限脚本设置为 1024。每次改变模型、数据、经验或阈值都使用新实验目录；`--resume` 只用于同配置续跑，代码更新仅可在了解影响后使用 `--allow-code-change`。旧进度记录来自不同路径、代码与配置，不承诺在整理包里直接续跑；其分析与指标重算仍可使用。

### 7.4 离线指标与通信实验

```bash
python -m experiment.test \
  --input experiment/qwen3_betterex_full2/test/progress.json \
  --expected-count 93
python -m experiment.testval \
  --input experiment/qwen_validation/test/progress.json
python benchmark_competition.py
```

第二条要求先完成对应验证实验。评测默认将 `recommended_drugs=null` 的失败例从质量分数中排除，并单独报告失败率；`exact_match_accuracy_all_recorded` 才是以所有记录病例为分母的准确率。不能省略分母和失败例处理。旧同病例经验诊断会刻意使用同病例经验，不能作为正常无泄漏泛化测试。

## 8. 已完成检查与仍需补足的部分

详细见 [docs/AUDIT.md](docs/AUDIT.md)。本次没有调用真实付费模型、没有下载模型、没有 GPU 训练/推理，因此只验证本地逻辑与工程可运行性，不把历史结果当作本次重跑结果。

数据来源及上游许可已核验（见 docs/DATA_PROVENANCE.md）。仍需作者补齐：模型精确版本或权重 revision、原环境 freeze、验证集阈值选择依据、失败样本一致计分、经验医学人工审核，以及患者级/语义重复检查。项目明确使用真实标签数量作为目标药物数，论文中应写明该 oracle-count 条件或补做数量预测实验。

## 9. 仓库内容与维护

公开仓库：https://github.com/197388tt-source/TinyHospital 。本仓库包含完整复现输入、最终经验、逐例结果和历史分析，源数据与实验结果已纳入版本控制。大于 200 KB 的 JSONL 日志和完整数据以 `.gz` 无损存储；准备脚本自动恢复原文件并校验哈希，预测 progress.json 和结果汇总保留可直接阅读的 JSON。去重后的运行副本由 `python scripts/prepare_assets.py` 恢复，生成文件由 `.gitignore` 排除，避免再次提交重复内容。

模型权重、真实密钥、虚拟环境、编译缓存和可重建的 SQLite 向量缓存不提交。真实模型推理需自行配置模型/API，离线结果复算和通信测试不需要真实 API。

数据来源已与 [DialMed 官方公开仓库](https://github.com/f-window/DialMed) 核验，并保留其 MIT 许可原文。逐文件核验与测试集的一处括号补全见 [数据来源说明](docs/DATA_PROVENANCE.md)。本项目原创代码尚未单独指定开源许可证。
