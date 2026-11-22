# 资源准备

`prepare_assets.py` 根据根目录 assets_manifest.json 恢复 6 个数据子集、52 个重复文件路径，以及 53 个无损压缩的原始数据/日志；`--check` 仅校验。它核对 SHA-256、检查越界路径，并拒绝覆盖已经修改的实验文件。

命令：`python scripts/prepare_assets.py`。运行目录是项目根目录。
