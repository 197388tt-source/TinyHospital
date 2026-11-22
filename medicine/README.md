# 药物知识

- `medicine.json`：允许推荐的候选药物名称。
- `medicine_introduction.json`：按药名检索的药品说明信息。

通过 `tool/medicine_store.py` 加载。Doctor 使用白名单约束输出，Expert 在经验提炼阶段使用说明书审核。测试阶段不调用 Expert 提炼经验。资料和模型输出均用于研究，不能视为针对真实患者的处方。
