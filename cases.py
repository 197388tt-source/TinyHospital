"""DialMed case loading and the known drug count supplied to Patient."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DATA_DIRECTORY = Path(__file__).resolve().parent / "data"
DIALMED_TRAIN_PATH = DATA_DIRECTORY / "traincut.txt"
DIALMED_TEST_PATH = DATA_DIRECTORY / "testcut.txt"
DIALMED_DEV_PATH = DATA_DIRECTORY / "devcut.txt"


def default_dataset_path(training: bool) -> Path:
    """Return the project dataset for the requested train/test stage."""
    if type(training) is not bool:
        raise ValueError("training 必须为布尔值")
    return DIALMED_TRAIN_PATH if training else DIALMED_TEST_PATH


def memory_source_dataset_path(dataset_path: Path, training: bool) -> Path:
    """Resolve held-out test/validation subsets to their training memory source."""
    path = Path(dataset_path)
    if not training:
        for suffix in ('cut', 'piyan', ''):
            if path.resolve() in {(DATA_DIRECTORY / f'{split}{suffix}.txt').resolve()
                                  for split in ('test', 'dev')}:
                return DATA_DIRECTORY / f'train{suffix}.txt'
    return path


@dataclass(frozen=True)
class DialMedCase:
    case_id: int
    patient_input: str
    drug_labels: tuple[str, ...]

    @property
    def drug_count(self) -> int:
        return len(self.drug_labels)


def load_dialmed_cases(path: Path = DIALMED_TRAIN_PATH, limit: int | None = None) -> list[DialMedCase]:
    if limit is not None and limit < 1:
        raise ValueError("病例数必须大于 0")
    cases = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            labels = raw.get("label")
            dialog = raw.get("dialog")
            if not isinstance(labels, list) or not labels or any(
                not isinstance(name, str) or not name.strip() for name in labels
            ) or not isinstance(dialog, list):
                raise ValueError(f"DIALMED 第 {line_number} 行格式错误")
            for utterance in dialog:
                if not isinstance(utterance, str):
                    raise ValueError(f"DIALMED 第 {line_number} 行 dialog 格式错误")
            if not dialog:
                raise ValueError(f"DIALMED 第 {line_number} 行 dialog 为空")
            patient_input = json.dumps(dialog, ensure_ascii=False)
            cases.append(DialMedCase(line_number, patient_input, tuple(labels)))
            if limit is not None and len(cases) >= limit:
                break
    if not cases or (limit is not None and len(cases) < limit):
        raise ValueError("DIALMED 病例数量不足")
    return cases
