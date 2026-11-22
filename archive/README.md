# 历史资料

`nested/` 保存原嵌套项目压缩包中独有的经验、实验日志及消融表。旧代码差异保存在 nested/code_differences.patch，不覆盖当前入口。历史实验的模型、提示词、阈值或选择策略可能不同；原配置以各自 progress.json 为准。

历史消融复算：先在根目录运行资源准备脚本，再运行 `python archive/nested/experiment/experience_ablation_analysis/traincut_leave_one_out/build_analysis.py`。4 个 CSV 与 summary.json 已验证可逐字节重现。
