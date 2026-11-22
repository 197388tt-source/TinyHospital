# 数据输入

仓库以 `.txt.gz` 无损保存完整数据，准备脚本恢复后，`train.txt`、`dev.txt`、`test.txt` 是一行一条记录的完整 JSONL；label 为药物标签，dialog 为遮盖用药建议后的输入，其他字段供数据来源追溯。主流程只把 dialog 编码，标签仅评判使用，但标签数量会作为 count 提供给 Doctor。

运行 `python scripts/prepare_assets.py` 恢复胃炎 `*cut.txt` 和皮炎 `*piyan.txt`。病例 ID 是子集物理行号，禁止重排行；恢复脚本保留原顺序和字节。统计、拆分与限制见[根 README](../README.md#5-数据去重与资源恢复)。

公开来源与许可见 [DATA_PROVENANCE.md](../docs/DATA_PROVENANCE.md)。
