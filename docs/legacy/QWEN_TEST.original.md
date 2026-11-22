> 历史文档：保留设计演进资料，命令、路径、模型及行为可能已过时。当前运行以根目录 README.md 和源码为准。

# Qwen3-8B Doctor 测试

使用 `agent/doctor_Qwen.py` 的测试提示词和本地 Qwen3-8B。Expert 保持原 API 客户端，测试阶段只评判结果，不生成新经验。Doctor 单次加载、关闭 thinking、贪心生成，默认最多生成 256 token，保留 Agent 原有最多 3 次格式/白名单校验尝试。

在 GPU 可用的节点运行：

```bash
conda activate tiny-hospital
cd /home/Users/licx/OS-Work/tiny_hospital

python main.py \
  --stage test \
  --runtime sync \
  --doctor-backend local-qwen \
  --memory-dir history \
  --experiment-dir experiment/qwen3_8b_history \
  --dataset data/testcut.txt \
  --count 93 \
  --chat-timeout 120
```

模型默认路径：`/home/Users/SharedFile/Models/Qwen/Qwen3-8B`。可用 `--doctor-model-path` 和 `--doctor-max-new-tokens` 调整。本地 Doctor 的训练和测试均使用 `--runtime sync`。

默认沿用 embedding 检索（API、Qwen3-Embedding-8B、相似度阈值 0.60），运行环境须配置 `EMBEDDING_API_KEY`；Doctor 本地推理不使用此密钥。检索仍只返回与当前病例匹配的历史经验，不保证每例同时包含正向经验和反思。

首次运行将 `history/golden_experience.json` 和 `history/reflection_memory.json` 原样复制到实验目录。不会修改来源经验；续跑使用冻结副本，不重新复制后来更新的 history。不要将实验目录设置为 history。

结果在 `experiment/qwen3_8b_history/test/progress.json` 的 `results` 中；该文件还保存模型配置、完成病例编号和经验快照。`llm_usage.jsonl` 记录 Doctor 的本地 tokenizer token 计数，`diagnostics.jsonl` 记录调用和校验诊断。

中断后使用相同命令并添加 `--resume`。更换提示词、配置或数据后请换新实验目录。可先用 `--count 3` 试跑，再用 `--count 93 --resume` 扩大范围。该参数表示从数据集开头选取的总病例数。

离线验证（不会加载大模型或请求 API）：

```bash
python -m unittest discover -s tests -p test_qwen_experiment.py
```

## Qwen 训练、DeepSeek 测试

```bash
bash train_qwen.sh
bash test_deepseek_after_qwen.sh
```

按顺序运行。两者共用新目录 `experiment/qwen_train_deepseek_test`：训练使用 `data/traincut.txt` 的 686 例，测试使用 `data/testcut.txt` 的 93 例。训练从空经验库开始，测试冻结并使用此次训练产生的经验，不复制 `history`。原有 `test_qwen.sh`、`test_deepseek.sh` 仍用于已有 history 经验的测试。

训练 Doctor 使用本地 Qwen3-8B，输出药名及 `reasoning_summary`，生成上限为 512 token；这里的训练是生成经验，不微调模型权重。Expert 使用 `tool/llm_client.py` 配置的 DeepSeek API。测试 Doctor 同样使用该 API；测试阶段 Expert 保留 API 客户端，但只做标签比较，不调用模型提炼经验。

脚本自动使用 tiny-hospital 的 Python，继承 `EMBEDDING_API_KEY` 和聊天 API 配置（可通过 `DLUT_API_KEY` 指定）。首次运行不加 `--resume`，中断后在相应脚本后追加 `--resume`。可追加 `--count 3` 先试跑，之后用默认数量加 `--resume` 继续。
