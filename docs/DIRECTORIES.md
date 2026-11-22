# 全部目录说明

表格覆盖整理包实体目录以及准备脚本将恢复的目录。运行产生的新目录不在本清单中。

| 目录 | 用途 |
| --- | --- |
| `.github/workflows/` | 自动离线复现检查，配置见 offline-checks.yml |
| `agent/` | 四角色、基类与输出验证 |
| `archive/` | 历史资料 |
| `archive/nested/` | 嵌套项目独有历史资料与旧代码差异，非当前入口 |
| `archive/nested/experiment/` | 嵌套项目独有历史资料与旧代码差异，非当前入口 |
| `archive/nested/experiment/empty_memory/` | 旧代码路线 empty_memory 的历史快照/日志；目录名不等于当前版本配置 |
| `archive/nested/experiment/experience/` | 旧路线的独立经验版本，不能与 betterex 混合 |
| `archive/nested/experiment/experience_ablation_analysis/` | 旧代码路线的经验消融表、汇总和离线分析脚本；先恢复资源 |
| `archive/nested/experiment/experience_ablation_analysis/traincut_leave_one_out/` | 旧代码路线的经验消融表、汇总和离线分析脚本；先恢复资源 |
| `archive/nested/experiment/experience_lcx/` | 旧路线的独立经验版本，不能与 betterex 混合 |
| `archive/nested/experiment/qwen_devcut_glm3_gpu0_empty/` | 旧代码路线 qwen_devcut_glm3_gpu0_empty 的历史快照/日志；目录名不等于当前版本配置 |
| `archive/nested/experiment/qwen_devcut_glm3_gpu0_empty/test/` | 旧代码路线 qwen_devcut_glm3_gpu0_empty 的历史快照/日志；目录名不等于当前版本配置 |
| `archive/nested/experiment/qwen_devcut_glm3_gpu0_one_each/` | 旧代码路线 qwen_devcut_glm3_gpu0_one_each 的历史快照/日志；目录名不等于当前版本配置 |
| `archive/nested/experiment/qwen_devcut_glm3_gpu0_one_each/test/` | 旧代码路线 qwen_devcut_glm3_gpu0_one_each 的历史快照/日志；目录名不等于当前版本配置 |
| `archive/nested/experiment/qwen_devcut_glm3_gpu0_positive_only/` | 旧代码路线 qwen_devcut_glm3_gpu0_positive_only 的历史快照/日志；目录名不等于当前版本配置 |
| `archive/nested/experiment/qwen_devcut_glm3_gpu0_positive_only/test/` | 旧代码路线 qwen_devcut_glm3_gpu0_positive_only 的历史快照/日志；目录名不等于当前版本配置 |
| `archive/nested/experiment/qwen_devcut_glm3_gpu0_reflection_only/` | 旧代码路线 qwen_devcut_glm3_gpu0_reflection_only 的历史快照/日志；目录名不等于当前版本配置 |
| `archive/nested/experiment/qwen_devcut_glm3_gpu0_reflection_only/test/` | 旧代码路线 qwen_devcut_glm3_gpu0_reflection_only 的历史快照/日志；目录名不等于当前版本配置 |
| `archive/nested/experiment/qwen_positive_only_memory/` | 旧代码路线 qwen_positive_only_memory 的历史快照/日志；目录名不等于当前版本配置 |
| `archive/nested/experiment/qwen_reflection_only_memory/` | 旧代码路线 qwen_reflection_only_memory 的历史快照/日志；目录名不等于当前版本配置 |
| `archive/nested/experiment/qwen_testcut_glm3_gpu0_experience_lcx/` | 旧代码路线 qwen_testcut_glm3_gpu0_experience_lcx 的历史快照/日志；目录名不等于当前版本配置 |
| `archive/nested/experiment/qwen_testcut_glm3_gpu0_experience_lcx/test/` | 旧代码路线 qwen_testcut_glm3_gpu0_experience_lcx 的历史快照/日志；目录名不等于当前版本配置 |
| `archive/nested/experiment/qwen_testcut_glm3_gpu0_experience_lcx_threshold_08/` | 旧代码路线 qwen_testcut_glm3_gpu0_experience_lcx_threshold_08 的历史快照/日志；目录名不等于当前版本配置 |
| `archive/nested/experiment/qwen_testcut_glm3_gpu0_experience_lcx_threshold_08/test/` | 旧代码路线 qwen_testcut_glm3_gpu0_experience_lcx_threshold_08 的历史快照/日志；目录名不等于当前版本配置 |
| `archive/nested/experiment/qwen_testcut_glm3_gpu0_filtered_experience/` | 旧代码路线 qwen_testcut_glm3_gpu0_filtered_experience 的历史快照/日志；目录名不等于当前版本配置 |
| `archive/nested/experiment/qwen_testcut_glm3_gpu0_filtered_experience/test/` | 旧代码路线 qwen_testcut_glm3_gpu0_filtered_experience 的历史快照/日志；目录名不等于当前版本配置 |
| `archive/nested/experiment/qwen_testcut_glm3_gpu0_filtered_experience_threshold_08/` | 旧代码路线 qwen_testcut_glm3_gpu0_filtered_experience_threshold_08 的历史快照/日志；目录名不等于当前版本配置 |
| `archive/nested/experiment/qwen_testcut_glm3_gpu0_filtered_experience_threshold_08/test/` | 旧代码路线 qwen_testcut_glm3_gpu0_filtered_experience_threshold_08 的历史快照/日志；目录名不等于当前版本配置 |
| `archive/nested/experiment/qwen_testcut_glm3_gpu0_full_threshold_08/` | 旧代码路线 qwen_testcut_glm3_gpu0_full_threshold_08 的历史快照/日志；目录名不等于当前版本配置 |
| `archive/nested/experiment/qwen_testcut_glm3_gpu0_full_threshold_08/test/` | 旧代码路线 qwen_testcut_glm3_gpu0_full_threshold_08 的历史快照/日志；目录名不等于当前版本配置 |
| `archive/nested/experiment/qwen_testcut_glm3_gpu0_one_each/` | 旧代码路线 qwen_testcut_glm3_gpu0_one_each 的历史快照/日志；目录名不等于当前版本配置 |
| `archive/nested/experiment/qwen_testcut_glm3_gpu0_one_each/test/` | 旧代码路线 qwen_testcut_glm3_gpu0_one_each 的历史快照/日志；目录名不等于当前版本配置 |
| `archive/nested/experiment/qwen_train_deepseek_full/` | 旧代码路线 qwen_train_deepseek_full 的历史快照/日志；目录名不等于当前版本配置 |
| `archive/nested/experiment/qwen_train_deepseek_full/train/` | 旧代码路线 qwen_train_deepseek_full 的历史快照/日志；目录名不等于当前版本配置 |
| `archive/nested/experiment/qwen_traincut_glm3_gpu0_empty/` | 旧代码路线 qwen_traincut_glm3_gpu0_empty 的历史快照/日志；目录名不等于当前版本配置 |
| `archive/nested/experiment/qwen_traincut_glm3_gpu0_empty/test/` | 旧代码路线 qwen_traincut_glm3_gpu0_empty 的历史快照/日志；目录名不等于当前版本配置 |
| `archive/nested/experiment/qwen_traincut_glm3_gpu0_leave_one_out/` | 旧代码路线 qwen_traincut_glm3_gpu0_leave_one_out 的历史快照/日志；目录名不等于当前版本配置 |
| `archive/nested/experiment/qwen_traincut_glm3_gpu0_leave_one_out/test/` | 旧代码路线 qwen_traincut_glm3_gpu0_leave_one_out 的历史快照/日志；目录名不等于当前版本配置 |
| `archive/nested/experiment/qwen_traincut_glm3_gpu0_positive_only/` | 旧代码路线 qwen_traincut_glm3_gpu0_positive_only 的历史快照/日志；目录名不等于当前版本配置 |
| `archive/nested/experiment/qwen_traincut_glm3_gpu0_positive_only/test/` | 旧代码路线 qwen_traincut_glm3_gpu0_positive_only 的历史快照/日志；目录名不等于当前版本配置 |
| `archive/nested/experiment/qwen_traincut_glm3_gpu0_reflection_only/` | 旧代码路线 qwen_traincut_glm3_gpu0_reflection_only 的历史快照/日志；目录名不等于当前版本配置 |
| `archive/nested/experiment/qwen_traincut_glm3_gpu0_reflection_only/test/` | 旧代码路线 qwen_traincut_glm3_gpu0_reflection_only 的历史快照/日志；目录名不等于当前版本配置 |
| `data/` | 完整数据；子集由索引恢复 |
| `docs/` | 审计与验证文档 |
| `docs/legacy/` | 已标注过时的设计资料 |
| `embedding/` | 编码、持久缓存、精确检索与向量传输 |
| `experiment/` | 评分/诊断代码及实验结果 |
| `experiment/betterex/` | 当前快照来源；686 例经验提炼训练；重复经验由准备脚本恢复 |
| `experiment/betterex/train/` | 当前快照来源；686 例经验提炼训练；逐例进度、诊断与通信/模型用量 |
| `experiment/qwen3_betterex_full2/` | 本地 Qwen + 当前经验的 93 例测试；重复经验由准备脚本恢复 |
| `experiment/qwen3_betterex_full2/test/` | 本地 Qwen + 当前经验的 93 例测试；逐例进度、诊断与通信/模型用量 |
| `experiment/qwen_trained_testnomemory/` | 本地 Qwen 无经验的 93 例对照；重复经验由准备脚本恢复 |
| `experiment/qwen_trained_testnomemory/test/` | 本地 Qwen 无经验的 93 例对照；逐例进度、诊断与通信/模型用量 |
| `experiment/vector_ab_binary/` | API Doctor 查询传输三模式的 10 例试验；重复经验由准备脚本恢复 |
| `experiment/vector_ab_binary/test/` | API Doctor 查询传输三模式的 10 例试验；逐例进度、诊断与通信/模型用量 |
| `experiment/vector_ab_shared/` | API Doctor 查询传输三模式的 10 例试验；重复经验由准备脚本恢复 |
| `experiment/vector_ab_shared/test/` | API Doctor 查询传输三模式的 10 例试验；逐例进度、诊断与通信/模型用量 |
| `experiment/vector_ab_text/` | API Doctor 查询传输三模式的 10 例试验；重复经验由准备脚本恢复 |
| `experiment/vector_ab_text/test/` | API Doctor 查询传输三模式的 10 例试验；逐例进度、诊断与通信/模型用量 |
| `experiment/vector_final_binary/` | API Doctor 查询传输三模式的 93 例试验；只用反思；重复经验由准备脚本恢复 |
| `experiment/vector_final_binary/test/` | API Doctor 查询传输三模式的 93 例试验；只用反思；逐例进度、诊断与通信/模型用量 |
| `experiment/vector_final_shared/` | API Doctor 查询传输三模式的 93 例试验；只用反思；重复经验由准备脚本恢复 |
| `experiment/vector_final_shared/test/` | API Doctor 查询传输三模式的 93 例试验；只用反思；逐例进度、诊断与通信/模型用量 |
| `experiment/vector_final_text/` | API Doctor 查询传输三模式的 93 例试验；只用反思；重复经验由准备脚本恢复 |
| `experiment/vector_final_text/test/` | API Doctor 查询传输三模式的 93 例试验；只用反思；逐例进度、诊断与通信/模型用量 |
| `history/` | 经验存取代码与旧经验 |
| `medicine/` | 药物白名单和说明书 |
| `memories/` | 默认及其他独立经验版本 |
| `memories/alternatives/` | 旧压缩包中的独立筛选经验（55 正向 / 493 反思） |
| `memories/alternatives/nested_filtered/` | 旧压缩包中的独立筛选经验（55 正向 / 493 反思） |
| `memories/final/` | 外层 betterex 的默认快照（52 正向 / 235 反思）与来源哈希 |
| `runs/` | 历史性能与通信结果 |
| `scripts/` | 资源准备工具 |
| `tests/` | 离线 unittest |
| `tool/` | 客户端、记录、重试及进度工具 |
