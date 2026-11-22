# 离线测试

| 文件 | 内容 |
| --- | --- |
| test_competition_benchmark.py | Offline regression checks for the competition-facing comparison report. |
| test_decision_vector.py | 测试 decision_vector |
| test_doctor_prompts.py | Stage-specific model inputs and output contracts, without live API calls. |
| test_doctor_qwen_prompts.py | Qwen Doctor prompt and response contracts, without model calls. |
| test_embedding.py | Dependency-free correctness checks; fake vectors do not measure semantics. |
| test_experiment_progress.py | 测试 experiment_progress |
| test_hospital_process.py | 测试 hospital_process |
| test_incremental_history.py | Stage 8: only the retrieval.history reply changes on the IPC wire. |
| test_incremental_medicines.py | Stage 8: only retrieval.medicines replies become incremental. |
| test_ipc_runtime.py | 测试 ipc_runtime |
| test_llm_client.py | 测试 llm_client |
| test_main.py | 测试 main |
| test_memory.py | 测试 memory |
| test_performance_baseline.py | 测试 performance_baseline |
| test_prepare_assets.py | 测试 prepare_assets |
| test_protocol.py | 测试 protocol |
| test_qwen_experiment.py | Offline checks for local Doctor routing and frozen test memories. |
| test_qwen_invalid_output.py | Invalid Qwen output is a completed failed case, not a fatal session error. |
| test_retrieval_policy.py | Threshold/diagnostic regression checks without model calls. |
| test_same_case_memory.py | Offline verification of same-case replay; never load a real model. |
| test_semantic_retrieval.py | Semantic routing tests with a local embeddings API; no external models or keys. |
| test_shared_medicines.py | Retrieval-owned medicine shared memory reaches Expert and is reclaimed. |
| test_shared_transport.py | 测试 shared_transport |
| test_stage5_recovery.py | 测试 stage5_recovery |
| test_structured_payload.py | Native TH-ACP data payload compatibility and shared field delivery. |

运行：`python -m unittest discover -s tests`。153 项中 152 通过、1 跳过（缺少旧冻结基线）；不依赖真实模型 API。
