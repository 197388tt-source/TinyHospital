# 验证记录

日期：2026-11-22。环境：Linux，Python 3.12.14，NumPy 2.3.5。

初次检查在整理压缩包的独立解压目录执行；加入 GitHub 无损压缩恢复支持后，在发布工程重新执行完整测试。以下区分原始压缩包检查与最终发布工程验证。

| 检查 | 结果 |
| --- | --- |
| ZIP 完整性 | `ZipFile.testzip()` 通过 |
| 资源恢复 | 111 个资源全部通过 SHA-256 校验并恢复（6 个子集 + 52 个重复文件路径 + 53 个无损压缩资源） |
| 全部输入数据 | 恢复后的 9 个数据集逐字节等于上传文件 |
| 最终经验 | 两份 JSON 逐字节等于外层 betterex；52/235 条 |
| 完整 unittest | **153 项：152 通过，1 跳过，0 失败，0 报错**，约 37.32 秒 |
| 跳过原因 | 原项目缺少 `runs/performance_baseline.json` 冻结基线 |
| Python 静态解析 | 76 个 `.py` 文件全部通过 AST 解析；活动代码 compileall 通过 |
| CLI 和启动脚本 | main、test、testval、benchmark help；4 个 shell 脚本 bash -n 通过 |
| 历史消融分析 | 重新执行旧项目 `traincut_leave_one_out/build_analysis.py`，4 个 CSV 和 1 个 summary.json 与原结果逐字节相同 |
| 指标复算 | 8 组外层测试结果离线重算，见 `recomputed_metrics.json` |
| 常见凭据扫描 | Python、文档、数据、历史 patch 及 Word XML 中未匹配到扫描规则中的真实密钥格式 |

测试使用模型替身/本地 HTTP 服务。未运行真实聊天/embedding API，未下载模型，未安装验证 GPU 依赖组合，没有医学人工审核。成功的测试不等于真实模型可达、结果可完全复现或医疗安全得到保证。

清理不重排数据，不改写经验正文，不修改历史模型预测。旧代码差异只作为档案保留。版本选择已由作者确认：外层 betterex 是最终版。
