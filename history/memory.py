"""Indexed retrieval of approved success lessons and error reflections."""

from __future__ import annotations

import json
from hashlib import blake2b, sha256
from pathlib import Path
from time import perf_counter

from cases import DIALMED_TRAIN_PATH
from tool.case_utils import _case_summary, _features
from tool.json_store import write_json_rows
from tool.diagnostics import emit

REFLECTION_MEMORY_PATH = Path(__file__).resolve().parent / 'reflection_memory.json'
GOLDEN_EXPERIENCE_PATH = Path(__file__).resolve().parent / 'golden_experience.json'


class CaseMemoryStore:
    """Keep reusable lessons and recover their source dialogue from training data."""

    MIN_SIMILARITY = 0.15
    EMBEDDING_MIN_SIMILARITY = 0.60  # Provisional conservative cutoff; calibrate on held-out pairs.
    MAX_RESULTS = 3

    def __init__(self, golden_path: Path, reflection_path: Path, training: bool,
                 source_dataset_path: Path = DIALMED_TRAIN_PATH,
                 retrieval_mode: str = 'embedding', embedding_options: dict | None = None) -> None:
        if retrieval_mode not in ('embedding', 'jaccard', 'none'):
            raise ValueError('retrieval_mode 必须为 embedding、jaccard 或 none')
        self.retrieval_mode = retrieval_mode
        options = dict(embedding_options or {})
        self.min_similarity = (options.get('min_similarity', self.EMBEDDING_MIN_SIMILARITY)
                               if retrieval_mode == 'embedding' else self.MIN_SIMILARITY)
        if (isinstance(self.min_similarity, bool) or not isinstance(self.min_similarity, (int, float))
                or not 0 <= self.min_similarity <= 1):
            raise ValueError('embedding 最低相似度必须在 0 到 1 之间')
        self.diagnostics_path = options.get('diagnostics_path')
        self.diagnostics_verbose = options.get('diagnostics_verbose', False)
        self.golden_path = Path(golden_path)
        self.reflection_path = Path(reflection_path)
        self.training = training
        self.source_dataset_path = Path(source_dataset_path)
        self._source_dialogs: dict[int, str] = {}
        self._source_version: tuple[int, int] | None = None
        self._golden: dict[int, str] = {}
        self._reflections: list[dict] = []
        self._index: list[tuple[str, dict, str, set[str], str]] = []
        self._versions: tuple[tuple | None, tuple | None] = (None, None)
        self._queries = self._hits = self._returned = 0
        self._latency_ms = self._similarity_sum = 0.0
        self.last_retrieval_stats: dict | None = None
        self.semantic = None
        if retrieval_mode == 'embedding':
            from embedding.retrieval import SemanticMemoryIndex
            self.semantic = SemanticMemoryIndex(self.golden_path, self.reflection_path,
                                                self.source_dataset_path, embedding_options)
        self._reload()

    @staticmethod
    def _version(path: Path) -> tuple[int, int] | None:
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None
        return stat.st_mtime_ns, stat.st_size

    @classmethod
    def _memory_version(cls, path: Path) -> tuple | None:
        version = cls._version(path)
        if version is None:
            return None
        # Filesystems can reuse mtime for rapid same-size writes.
        return (*version, blake2b(path.read_bytes(), digest_size=8).digest())

    @staticmethod
    def _read(path: Path) -> list[dict]:
        if not path.exists():
            return []
        contents = path.read_text(encoding='utf-8').strip()
        if not contents:
            return []
        value = json.loads(contents)
        if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
            raise ValueError(f'记忆文件格式错误：{path}')
        return value

    @staticmethod
    def _dialog_key(dialog: str) -> str:
        try:
            utterances = json.loads(dialog)
        except (TypeError, ValueError):
            pass
        else:
            if isinstance(utterances, list) and all(isinstance(item, str) for item in utterances):
                return ' '.join(' '.join(utterances).casefold().split())
        return ' '.join(dialog.casefold().split())

    @staticmethod
    def _normalize_reflection(row: dict) -> dict:
        # Older six-field rows remain readable and are compacted on the next write.
        reflection = row.get('reflection') or row.get('lesson')
        if (type(row.get('case_id')) is not int or not isinstance(reflection, str)
                or not 0 < len(reflection.strip()) <= 250):
            raise ValueError('错误反思须包含 case_id 和250字以内的非空 reflection')
        return {'case_id': row['case_id'], 'reflection': reflection.strip()}

    def _load_source_dialogs(self) -> None:
        version = self._version(self.source_dataset_path)
        if version == self._source_version:
            return
        self._source_dialogs = {}
        if version is not None:
            with self.source_dataset_path.open(encoding='utf-8') as source:
                for case_id, line in enumerate(source, 1):
                    if not line.strip():
                        continue
                    dialog = json.loads(line).get('dialog')
                    if isinstance(dialog, list) and all(isinstance(item, str) for item in dialog):
                        self._source_dialogs[case_id] = json.dumps(dialog, ensure_ascii=False)
        self._source_version = version

    def _reload(self) -> None:
        self._load_source_dialogs()
        self._golden = {}
        for row in self._read(self.golden_path):
            if (set(row) != {'case_id', 'experience'} or type(row['case_id']) is not int
                    or not isinstance(row['experience'], str) or not row['experience'].strip()
                    or len(row['experience']) > 200 or row['case_id'] in self._golden):
                raise ValueError(f'成功经验文件格式错误：{self.golden_path}')
            self._golden[row['case_id']] = row['experience'].strip()
        self._reflections = [self._normalize_reflection(row)
                             for row in self._read(self.reflection_path)]
        if len({row['case_id'] for row in self._reflections}) != len(self._reflections):
            raise ValueError(f'错误反思文件包含重复 case_id：{self.reflection_path}')
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self._index = []
        for case_id, experience in self._golden.items():
            dialog = self._source_dialogs.get(case_id)
            if dialog:
                row = {'case_id': case_id, 'experience': experience}
                self._index.append(('golden_experiences', row, self._dialog_key(dialog),
                                    _features(dialog), dialog))
        for row in self._reflections:
            dialog = self._source_dialogs.get(row['case_id'])
            if dialog:
                self._index.append(('reflections', row, self._dialog_key(dialog),
                                    _features(dialog), dialog))
        self._versions = (self._memory_version(self.golden_path),
                          self._memory_version(self.reflection_path))
        if self.semantic is not None:
            self.semantic.synced = False

    def _sync_semantic(self) -> None:
        if self.semantic is not None and not self.semantic.synced:
            self.semantic.sync(self._index)

    def _refresh_if_changed(self) -> None:
        # Test keeps the files and training dialogue as they were at session startup.
        if self.training and (self._version(self.source_dataset_path) != self._source_version
                or (self._memory_version(self.golden_path),
                    self._memory_version(self.reflection_path)) != self._versions):
            self._reload()

    def retrieve(self, dialog: str | None, case_id: int, dataset_path: str,
                 limit: int = MAX_RESULTS, min_similarity: float | None = None,
                 query_vector=None, query_text_hash: str | None = None) -> dict:
        if min_similarity is None:
            min_similarity = self.min_similarity
        if dialog is None and query_vector is None:
            raise ValueError('检索必须提供对话文本或查询向量')
        if dialog is None and self.retrieval_mode != 'embedding':
            raise ValueError('无对话文本的检索仅支持 embedding 模式')
        if query_vector is not None and self.semantic is None:
            raise ValueError('只有 embedding 检索接受查询向量')
        if type(limit) is not int or limit < 0:
            raise ValueError('检索返回数量必须是非负整数')
        if (not isinstance(min_similarity, (int, float)) or isinstance(min_similarity, bool)
                or not 0 <= min_similarity <= 1):
            raise ValueError('最低相似度必须在 0 到 1 之间')
        from embedding.retrieval import dialog_text
        started = perf_counter()
        self._refresh_if_changed()
        self._sync_semantic()
        semantic_scores = self.semantic.scores(dialog, query_vector) if self.semantic is not None and limit else {}
        query, query_key = (_features(dialog), self._dialog_key(dialog)) if dialog is not None else (set(), None)
        ranked, candidates = [], []
        same_dataset = Path(dataset_path).resolve() == self.source_dataset_path.resolve()
        for bucket, row, old_key, old_features, old_dialog in self._index:
            union = query | old_features
            score = (semantic_scores.get(self.semantic.identity(bucket, row), 0.0)
                     if self.semantic is not None else
                     len(query & old_features) / len(union) if union else 0.0)
            reason = ('disabled' if self.retrieval_mode == 'none' else
                      'limit_zero' if limit == 0 else
                      'current_case' if row['case_id'] == case_id and same_dataset else
                      'identical_dialog' if (query_key is not None and old_key == query_key) or (
                          query_text_hash is not None and sha256(
                              dialog_text(old_dialog).encode()
                          ).hexdigest() == query_text_hash) else
                      'below_threshold' if score < min_similarity or score <= 0 else 'eligible')
            candidate = {'source_case_id': row['case_id'], 'kind': bucket,
                         'similarity': round(score, 6), 'reason': reason}
            candidates.append(candidate)
            if reason == 'eligible':
                ranked.append((score, bucket, row, old_key, old_dialog, candidate))
        ranked.sort(key=lambda item: (-item[0], item[1] != 'golden_experiences',
                                      item[2]['case_id']))
        result = {'golden_experiences': [], 'reflections': []}
        seen, similarities = set(), []
        for score, bucket, row, old_key, old_dialog, candidate in ranked:
            if old_key in seen:
                candidate['reason'] = 'duplicate_dialog'
                continue
            if len(similarities) >= limit:
                candidate['reason'] = 'outside_top_k'
                continue
            candidate['reason'] = 'selected'
            seen.add(old_key)
            item = {
                'case_id': row['case_id'],
                'summary': _case_summary(old_dialog),
                'dialog_excerpt': old_dialog[:200],
                'similarity': round(score, 4),
            }
            if bucket == 'golden_experiences':
                item['experience'] = row['experience']
            else:
                item['reflection'] = row['reflection']
            result[bucket].append(item)
            similarities.append(score)
        elapsed = (perf_counter() - started) * 1000
        self._queries += 1
        self._hits += bool(similarities)
        self._returned += len(similarities)
        self._latency_ms += elapsed
        self._similarity_sum += sum(similarities)
        self.last_retrieval_stats = {
            'retrieval_mode': self.retrieval_mode,
            'case_id': case_id, 'hit': bool(similarities), 'latency_ms': round(elapsed, 3),
            'returned_count': len(similarities),
            'similarities': [round(score, 4) for score in similarities],
            'min_similarity': min_similarity, 'limit': limit,
            'source_dataset': str(self.source_dataset_path.resolve()),
            'candidates': candidates,
        }
        emit(self.diagnostics_path, {'event': 'retrieval', **self.last_retrieval_stats})
        if self.diagnostics_verbose:
            print(f"[检索] 病例 {case_id}：{self.retrieval_mode}，阈值 {min_similarity}，"
                  f"候选 {len(candidates)}，返回 {len(similarities)} 条。", flush=True)
            for candidate in sorted(candidates, key=lambda item: -item['similarity'])[:5]:
                print(f"[检索候选] 来源病例 {candidate['source_case_id']} "
                      f"{candidate['kind']} score={candidate['similarity']:.4f} "
                      f"reason={candidate['reason']}", flush=True)
        return result

    @property
    def retrieval_statistics(self) -> dict:
        return {
            'retrieval_mode': self.retrieval_mode,
            'embedding': dict(self.semantic.stats) if self.semantic is not None else None,
            'min_similarity': self.min_similarity,
            'queries': self._queries, 'hits': self._hits,
            'hit_rate': self._hits / self._queries if self._queries else 0.0,
            'returned_count': self._returned,
            'average_latency_ms': round(self._latency_ms / self._queries, 3) if self._queries else 0.0,
            'average_returned_count': self._returned / self._queries if self._queries else 0.0,
            'average_similarity': self._similarity_sum / self._returned if self._returned else 0.0,
        }

    def _write_golden(self) -> None:
        write_json_rows(self.golden_path, [
            {'case_id': case_id, 'experience': experience}
            for case_id, experience in sorted(self._golden.items())
        ])

    def _write_reflections(self) -> None:
        write_json_rows(self.reflection_path, self._reflections)

    def save_success(self, case_id: int, experience: str | None) -> None:
        """Record only an approved lesson; clear stale lessons and reflections."""
        if not self.training:
            return
        if experience is not None and (not isinstance(experience, str)
                                      or not experience.strip() or len(experience.strip()) > 200):
            raise ValueError('成功经验须为200字以内的非空文字')
        self._refresh_if_changed()
        had_golden = case_id in self._golden
        self._golden.pop(case_id, None)
        if experience is not None:
            self._golden[case_id] = experience.strip()
        if had_golden or experience is not None:
            self._write_golden()
        old_size = len(self._reflections)
        self._reflections = [row for row in self._reflections if row.get('case_id') != case_id]
        if len(self._reflections) != old_size:
            self._write_reflections()
        self._rebuild_index()

        self._sync_semantic()

    def save_reflection(self, case_id: int, reflection: str) -> None:
        if not self.training:
            return
        row = self._normalize_reflection({'case_id': case_id, 'reflection': reflection})
        self._refresh_if_changed()
        self._reflections = [row for row in self._reflections if row.get('case_id') != case_id]
        self._reflections.append(row)
        self._write_reflections()
        if case_id in self._golden:
            del self._golden[case_id]
            self._write_golden()
        self._rebuild_index()
        self._sync_semantic()
