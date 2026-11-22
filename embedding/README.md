# 语义向量编码、检索与协议载荷

本模块已接入 Agent 主流程：来源病例对话编码、向量检索及 Patient → Retrieval 查询向量传输。经验正文是命中后的载荷，不参与主流程的相似度计算。
支持真实本地模型和 embeddings API；不提供伪语义后端。
详见 [向量传输说明](VECTOR_TRANSPORT.md)。

## 环境与模型

Python 3.10+。API 后端、数据读取和 SQLite 存储使用标准库，精确余弦检索使用 NumPy 连续 `float32` 矩阵。仅使用 API embedding 时安装：

```bash
python -m pip install -r embedding/requirements.txt
```

本地模型后端需要额外安装依赖（建议在独立环境安装，避免影响原实验）：

```bash
python -m pip install -r embedding/requirements-local.txt
```

必须选择一个适合中文的真正 embedding 模型，而不是普通聊天模型。
本地模型应是 SentenceTransformer 可加载的模型目录或模型 ID。
默认 `--device cpu`、只读取本地模型、不下载权重、不执行远端自定义代码。
需要下载时显式加 `--allow-download`；远端模型建议使用 `--revision` 固定提交。
`--model-version` 必填，应是实际权重的版本/提交或服务部署版本。
本地文件或 API 背后模型变化后，必须更新该值；程序无法发现服务商静默更换权重。
API 地址必须是完整的 embeddings URL，不会自动拼接路径，也不会复用原聊天密钥。

先用最小脚本测试 API，不会写入向量库：

```bash
export EMBEDDING_API_KEY='有效的embedding密钥'
python -m embedding.test_api \
  --endpoint https://你的网关/v1/embeddings \
  --model 实际的embedding模型名
```

成功时会打印 `ok`、模型名、向量维度和前五个数值；出现 `HTTP 404` 通常表示网关没有该路径，
`HTTP 400` 通常表示模型名或请求格式不匹配。脚本不会打印密钥，也不会打印完整响应。

本地后端接口依据 [Sentence Transformers 文档](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html)。
模型默认 prompt 被关闭；若模型要求 `query:` 等前缀，须按模型说明通过 `--prefix` 显式指定。
第一版同一配置仅支持统一前缀；不自动区分 query/document 两种任务指令。

## 运行目录

以下命令都在这里运行：

```bash
cd /data/liust/caozuoxitong/tiny_hospital
```

### 本地模型：先编码 10 条训练对话

替换模型目录与版本占位值后执行：

```bash
python -m embedding.build_vectors build \
  --source data/traincut.txt --kind dialog --namespace train-dialog \
  --backend local --model /实际的embedding模型目录 \
  --model-version 实际模型版本 \
  --output vectors/local.sqlite --limit 10
```

重复运行，预期 `encoded_texts` 为 0；去掉 `--limit 10` 后导入全量，只编码尚未缓存的文本。
需要 GPU 时增加 `--device cuda`。CPU 与 GPU 配置分开缓存，避免不明确的数值差异。

### API 后端

网关必须支持请求 `model/input/encoding_format`，并返回带 `index/embedding` 的 `data` 数组。
不能因为聊天接口能调用就假定 embeddings 接口存在。

```bash
export EMBEDDING_API_KEY='有效的embedding密钥'
python -m embedding.build_vectors build \
  --source data/traincut.txt --kind dialog --namespace train-dialog \
  --backend api --model 实际的embedding模型名 \
  --model-version 实际服务模型版本 \
  --endpoint https://你的网关/v1/embeddings \
  --output vectors/api.sqlite --limit 10
```

文本会发送给指定网关。密钥仅从环境读取，不保存到向量库或配置指纹；HTTP 错误不打印响应正文或密钥。
此模块不自动重试 API 请求；失败时数据库事务回滚，但已发生的 API 调用成本无法回滚。

### 成功经验、专家反思与测试集

使用上述相同模型参数，替换输入及 namespace：

| source | kind | namespace |
|---|---|---|
| `history/golden_experience.json` | `experience` | `golden-experience` |
| `history/reflection_memory.json` | `reflection` | `reflection` |
| `data/testcut.txt` | `dialog` | `test-query` |

空经验数组是合法输入，不会触发模型调用。原始 JSON 文件不会被改写。
测试集编码后仍是独立命名空间；后续检索必须只使用历史命名空间，不能搜索所有记录。
通用文本支持 `--kind text`，源文件为 JSONL，每行例如 `{"id":"stable-id","text":"文本"}`。

## 文本处理与长对话

- 对话只读取 `dialog`，按发言顺序用换行连接，保留角色、否定、数字及 `[MASK]`。
- 不读取 `label`、`disease`、`original_dialog` 作为 embedding 输入。
- 经验只读取 `experience`；反思只读取 `reflection`。
- NFC 规范化、统一换行、去掉空行及每行两端空白，不删除医学细节。
- 默认按 256 个 Unicode 字符非重叠分段，覆盖全部文本。
- 各段 L2 归一化，按原分段字符数加权平均，最终再次 L2 归一化。
- 本地后端编码前检查 token 长度，超过模型限制直接报错，不静默截断；减小 `--chunk-chars` 并改用新向量库。
- 字符分段加权聚合是一种可复现的初版策略，不等同于模型完整阅读整段对话，后续需评估长文本质量。

## 增量更新和存储

输出是独立 SQLite 文件，向量以 little-endian float32 BLOB 存储。
选择 SQLite 而不是分别写矩阵和索引文件，是为了使文本映射、向量和增量更新可以原子提交。
原文与来源路径也保存在库中，便于追溯；它不是匿名数据文件。

- 首次运行：编码所有未缓存文本。
- 重复运行：根据预处理后文本 SHA-256 复用缓存，不调用模型。
- 新增记录：只编码新文本；相同内容即使 ID 不同也可复用向量。
- 修改记录：更新映射，只重算变化文本。
- 默认不删除未出现在当前输入中的记录，支持分批导入。
- `--prune`：将指定 namespace 与完整源文件同步，移除缺失记录；其他 namespace 不受影响。
- 禁止 `--prune` 与 `--limit` 同用；向量缓存中未再引用的旧向量仍保留，不能将缓存表直接作为检索候选。
- 不后台监听文件；修改源文件后需重新执行 build 或调用 `sync()`。
- 模型、版本、前缀、分段、设备或预处理配置不同，拒绝写入同一个库，请使用新输出路径。
- 导入期间单写者事务；适用于当前离线构建。其他写入最多等待 30 秒，并行大规模导入不是本阶段目标。

ID 优先读取 `record_id`、`memory_id`、`id`。当前 DIALMED 无独立 ID，沿用原系统的物理行号；
当前经验文件回退到 `case_id`。若重新排序/插入数据集行，应全量 `--prune` 同步，不能认为旧行号还是同一个病例。
生产数据推荐上游提供稳定 ID。

查看库摘要：

```bash
python -m embedding.build_vectors inspect --output vectors/local.sqlite
```

重点字段：`added`、`updated`、`unchanged`、`encoded_texts`、`reused_records`、`deleted`、
`dimension`、`new_vector_bytes`、`encoding_ms`、`total_ms`。
`backend_calls` 是编码批次数，`input_characters` 是编码前的文本字符数，不是 API Token。
`new_vector_bytes` 只计本轮新向量的数值载荷，不是 SQLite 文件大小或实际网络传输量。
`total_ms` 为 sync 内部耗时，包含首次模型加载（如发生），不包含源文件读取和最后事务提交。

## 程序接口

```python
from pathlib import Path
from embedding import EmbeddingConfig, Vectorizer, VectorStore, load_records

config = EmbeddingConfig('local', '/实际的embedding模型目录', '实际模型版本')
encoder = Vectorizer(config)

# 当前对话：只编码，不写历史库。没有缓存时每次会计算。
query_vectors = encoder.encode_texts(['患者：最近入睡困难\n医生：持续多久了？'])

# 历史对话：导入并缓存。模型配置必须与构建时完全一致。
records = load_records(Path('data/traincut.txt'), 'dialog')
with VectorStore(Path('vectors/local.sqlite'), config) as store:
    stats = store.sync('train-dialog', records, encoder)
    item = store.get('train-dialog', '1')
    vector = item['vector']
```

本模块不修改模型权重；train/test 表示数据用途，不是训练 embedding 模型。
原来的 `main.py` 和通信消融仍走原路径，保持前后对照的基线。
主流程默认使用 embedding，可用 `--retrieval-mode jaccard` 保持原检索方式。独立 build 命令仅建库；主流程经验索引使用独立的 semantic-memory.sqlite。
主流程的经验检索只将经验所属的来源病例对话写入语义索引；经验和反思正文作为命中后的返回载荷，不参与余弦相似度计算。上文的 `--kind experience` / `--kind reflection` 仍是通用离线建库能力，不代表主流程使用经验正文来匹配病例。
同步完成后，主流程按稳定记录顺序将所有已归一化的历史对话向量装入一个连续的 NumPy `float32` 矩阵。每次查询通过一次 `matrix @ query` 穷举计算全部余弦相似度，不采样、不分组，也不使用近似最近邻。运行统计中的 `search_backend=numpy_exact`、`matrix_rows`、`matrix_dimension`、`matrix_bytes`、`searches` 和 `search_ms` 可用于评估实际规模下的检索开销。

## 功能测试

无需下载模型、无需 API、无需安装本地模型依赖：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p test_embedding.py -v
```

测试使用专用模拟编码器，只验证接口、全量文本分段、标签隔离、增量缓存、事务回滚、
float32 读写、配置隔离、API 返回排序等，不证明真实语义质量或 Token 节省。
首次使用真实模型后，另做少量相关/无关文本的语义检查，再开展正式检索和性能对照。
