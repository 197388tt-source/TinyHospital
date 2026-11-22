# 实验结果总览

以下为已提交逐例预测的离线复算结果，并非本次重新调用真实模型得到的结果。

| 实验 | 病例 | 失败 | micro F1（排除失败） | 全部记录完全匹配 | 预测来源 |
| --- | ---: | ---: | ---: | ---: | --- |
| qwen3_betterex_full2 | 93 | 1 | 54.3956% | 27.9570% | [progress.json](../experiment/qwen3_betterex_full2/test/progress.json) |
| qwen_trained_testnomemory | 93 | 2 | 53.5912% | 27.9570% | [progress.json](../experiment/qwen_trained_testnomemory/test/progress.json) |
| vector_ab_binary | 10 | 0 | 62.5000% | 40.0000% | [progress.json](../experiment/vector_ab_binary/test/progress.json) |
| vector_ab_shared | 10 | 0 | 56.2500% | 30.0000% | [progress.json](../experiment/vector_ab_shared/test/progress.json) |
| vector_ab_text | 10 | 0 | 56.2500% | 30.0000% | [progress.json](../experiment/vector_ab_text/test/progress.json) |
| vector_final_binary | 93 | 0 | 55.7377% | 31.1828% | [progress.json](../experiment/vector_final_binary/test/progress.json) |
| vector_final_shared | 93 | 0 | 59.5628% | 37.6344% | [progress.json](../experiment/vector_final_shared/test/progress.json) |
| vector_final_text | 93 | 0 | 54.0984% | 31.1828% | [progress.json](../experiment/vector_final_text/test/progress.json) |

## 复算命令

```bash
python scripts/prepare_assets.py
python -m experiment.test --input experiment/qwen3_betterex_full2/test/progress.json --expected-count 93
```

## 如何解读

- 最终经验为 betterex：52 条成功经验、235 条反思。训练过程是经验提炼，不更新模型权重。
- Qwen 有经验/无经验两组 micro F1 约相差 0.80 个百分点；失败数不同，没有完成显著性检验，不能直接认定因果提升。
- vector 三模式来自另一配置路线；真实模型输出不完全相同，不宜把质量差异归因于传输方式。
- null 预测被排除出 micro 指标；失败率和全部记录准确率应同时报告。
- docs/AUDIT.md 说明其他实验限制；archive/nested/ 保留旧路线与消融分析，不能与当前版本混为一组。
