"""Load the medication allow-list and disease-linked package inserts."""

from __future__ import annotations

import json
from pathlib import Path


MEDICINE_ROOT = Path(__file__).resolve().parents[1] / "medicine"


class MedicineKnowledgeBase:
    def __init__(
        self,
        catalog_path: Path = MEDICINE_ROOT / "medicine.json",
        introduction_path: Path = MEDICINE_ROOT / "medicine_introduction.json",
    ) -> None:
        self.catalog_path = catalog_path
        self.introduction_path = introduction_path
        names = json.loads(catalog_path.read_text(encoding="utf-8"))
        if isinstance(names, dict):
            if any(type(index) is not int for index in names.values()) or sorted(names.values()) != list(range(len(names))):
                raise ValueError(f"药物白名单格式错误：{catalog_path}")
            names = list(names)
        if not isinstance(names, list) or not names or any(
            not isinstance(name, str) or not name.strip() for name in names
        ) or len(names) != len(set(names)):
            raise ValueError(f"药物白名单格式错误：{catalog_path}")
        self.names = tuple(names)
        self._allowed = set(names)

        records = json.loads(introduction_path.read_text(encoding="utf-8")) if introduction_path.exists() else []
        if not isinstance(records, list) or any(
            not isinstance(item, dict)
            or not isinstance(item.get("通用名称"), str)
            or not isinstance(item.get("对应疾病"), (str, type(None)))
            for item in records
        ):
            raise ValueError(f"药品说明书文件格式错误：{introduction_path}")
        self._records = [item for item in records if item["通用名称"] in self._allowed]

    def related_to(self, diseases: list[str], limit: int = 12) -> list[dict[str, str]]:
        """Return distinct whitelist medicines whose disease metadata matches candidates."""
        results: list[dict[str, str]] = []
        seen: set[str] = set()
        for disease in diseases:
            normalized_disease = disease
            for qualifier in ("慢性", "急性", "亚急性", "反复性", "老年人", "小儿", "儿童", "过敏性"):
                normalized_disease = normalized_disease.replace(qualifier, "")
            for record in self._records:
                mapped_disease = record.get("对应疾病") or ""
                medicine_name = record["通用名称"]
                indication = record.get("适应症") or ""
                lookup_text = f"{mapped_disease}{medicine_name}{indication}"
                disease_field_match = bool(
                    mapped_disease and len(mapped_disease) >= 3 and (
                        mapped_disease == disease
                        or disease in mapped_disease
                        or mapped_disease in disease
                    )
                )
                # Metadata disease matches rank naturally by source order. The normalized
                # disease term also lets a condition such as 慢性鼻炎 match 鼻炎 products.
                if not disease_field_match and not (normalized_disease and normalized_disease in lookup_text):
                    continue
                name = medicine_name
                if name in seen:
                    continue
                seen.add(name)
                item = {
                    "medicine": name,
                    "matched_disease": mapped_disease if disease_field_match else disease,
                    "match_source": "对应疾病" if disease_field_match else "药名或适应症",
                }
                for source, target in (
                    ("适应症", "indications"), ("禁忌", "contraindications"),
                    ("注意事项", "precautions"), ("药物相互作用", "interactions"),
                ):
                    value = record.get(source)
                    if isinstance(value, str) and value.strip():
                        item[target] = value.strip()
                results.append(item)
                if len(results) >= limit:
                    return results
        return results

    def introductions_for(self, names: list[str]) -> dict[str, dict]:
        """Return short package insert excerpts for post-recommendation review."""
        by_name = {row["通用名称"]: row for row in self._records}
        result = {}
        for name in names:
            row = by_name.get(name)
            if row is None:
                result[name] = {"说明": "药物说明文件中暂无该药物的记录"}
                continue
            summary = {}
            for field in ("适应症", "禁忌", "注意事项", "不良反应"):
                value = row.get(field)
                if isinstance(value, str) and value.strip():
                    summary[field] = value.strip()[:350]
            result[name] = summary or {"说明": "该条药物说明缺少可用内容"}
        return result
