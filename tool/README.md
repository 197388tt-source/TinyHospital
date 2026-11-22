# 通用工具

| 文件 | 内容 |
| --- | --- |
| __init__.py | Shared helpers for model access, local data, storage, and tracing. |
| api_retry.py | Bounded retries for transient transport errors; never log credentials. |
| case_utils.py | Dialog features and compact case summaries. |
| diagnostics.py | Append diagnostics; golden-experience review events include model replies, never API credentials. |
| experiment_progress.py | Atomic per-case results and committed memory snapshots for restart. |
| json_store.py | Atomic writes for JSON row stores. |
| llm_client.py | Minimal OpenAI-compatible chat client for the school model gateway. |
| llm_usage.py | Append model token totals without generating case reports. |
| local_qwen_client.py | Local Qwen3 non-thinking chat adapter for the Doctor agent. |
| medicine_store.py | Load the medication allow-list and disease-linked package inserts. |
| tracing.py | Communication trace and per-case packet usage statistics. |
