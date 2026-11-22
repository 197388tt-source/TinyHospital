# Agent 代码

| 文件 | 职责 |
| --- | --- |
| base.py | Agent 状态、请求发送、模型调用记录 |
| patient.py | 病例输入、标签分流、目标药物数、查询向量生成 |
| retrieval.py | 经验检索和药物说明获取；支持向量输入 |
| doctor.py | API 医生推荐，训练理由和测试药名输出 |
| doctor_Qwen.py | 复用共同训练提示词，本地 Qwen 格式及失败处理 |
| expert.py | 标签评估、成功经验审查与错误反思提炼 |
| validation.py | JSON、理由、证据引用和经验格式校验 |
| __init__.py | 对外导出角色 |

流程与可见信息见[根 README](../README.md#2-四个-agent-如何协作)。
