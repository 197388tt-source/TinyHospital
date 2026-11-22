# 数据来源与逐文件核验

数据来自 [DialMed 官方公开仓库](https://github.com/f-window/DialMed)，该仓库由 [COLING 2022 论文页面](https://aclanthology.org/2022.coling-1.60/) 直接链接。

上游核验 commit：`4cd19719ad8a25e0d2cb411cdede3aea9d9e095f`。

| 划分 | 病例数 | 本项目与公开来源的差异 |
| --- | ---: | --- |
| train | 9605 | 逐字节相同 |
| dev | 1192 | 逐字节相同 |
| test | 1199 | 1198 行完全相同；第 78 行仅在 dialog 和 original_dialog 的同一句中为 `[MASK！` 补上 `]`，没有新增患者信息 |

胃炎和皮炎子集由上述数据的原行构成，索引保留在 assets_manifest.json；最终经验、预测和消融分析是对这些公开基准病例的模型实验产物，不是新增采集的私人患者病历。病例编号仅为数据集物理行号。

上游 MIT 许可原文见 [third_party/DialMed.LICENSE](../third_party/DialMed.LICENSE)。本文件记录上游来源及既有许可，不替其他来源材料虚构授权。本项目原创代码尚未单独指定开源许可证。哈希和机器可读核验记录见 [data_provenance.json](data_provenance.json)。

引用：He et al. (2022), *DialMed: A Dataset for Dialogue-based Medication Recommendation*, COLING 2022, pp. 721–733。
