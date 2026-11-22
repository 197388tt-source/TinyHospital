# 经验管理代码与旧经验

`memory.py` 负责经验读取、更新、训练/测试隔离、检索阈值与 top-k，`__init__.py` 提供导出。这里的两份 JSON 是旧快照（19 条正向 / 149 条反思），保留用于历史可追溯。

正式最终快照在 [`memories/final/`](../memories/final/)，不要把本目录当作最终经验来源。启动命令显式使用 `--memory-dir memories/final`。
