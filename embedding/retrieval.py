"""Persistent semantic memory index, shared by synchronous and process sessions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .encoder import EmbeddingConfig, Vectorizer, normalize
from .records import TextRecord, clean_text
from .store import VectorStore


def dialog_text(dialog: str) -> str:
    """Use the same newline representation as the offline dialog encoder."""
    try:
        parts = json.loads(dialog)
    except (ValueError, TypeError):
        parts = None
    if isinstance(parts, list) and all(isinstance(part, str) for part in parts):
        dialog = '\n'.join(parts)
    return clean_text(dialog)


class SemanticMemoryIndex:
    def __init__(self, golden: Path, reflection: Path, dataset: Path, options=None):
        options = dict(options or {})
        config = options.get('config') or EmbeddingConfig(
            'api', 'Qwen3-Embedding-8B', 'Qwen3-Embedding-8B',
            endpoint='http://aigw.dlut.edu.cn/v1/embeddings')
        if isinstance(config, dict):
            config = EmbeddingConfig(**config)
        self.encoder = Vectorizer(config, batch_size=options.get('batch_size', 32),
                                 api_key_env=options.get('api_key_env', 'EMBEDDING_API_KEY'),
                                 timeout=options.get('timeout', 120))
        self.path = Path(options.get('path') or golden.parent / 'semantic-memory.sqlite')
        scope = json.dumps([str(p.resolve()) for p in (golden, reflection, dataset)])
        self.namespace = 'memory-' + hashlib.sha256(scope.encode()).hexdigest()[:24]
        self.identities: tuple[str, ...] = ()
        self.matrix = np.empty((0, 0), dtype=np.float32)
        self.synced = False
        self.query_cache = {}
        self.stats = {'syncs': 0, 'encoded_texts': 0, 'backend_calls': 0,
                      'query_encodings': 0, 'query_cache_hits': 0, 'encoding_ms': 0.0,
                      'search_backend': 'numpy_exact', 'searches': 0, 'search_ms': 0.0,
                      'matrix_rows': 0, 'matrix_dimension': 0, 'matrix_bytes': 0}

    @staticmethod
    def identity(bucket, row):
        return f"{bucket}:{row['case_id']}"

    def sync(self, entries):
        records = []
        for bucket, row, _, _, dialog in entries:
            # Similarity answers "which historical case resembles this case?".
            # Keep the lesson/reflection as the payload returned after a hit;
            # it must not change the vector used to match the two dialogues.
            text = dialog_text(dialog)
            records.append(TextRecord(self.identity(bucket, row), text, 'dialog',
                                      self.namespace))
        with VectorStore(self.path, self.encoder.config) as store:
            stats = store.sync(self.namespace, records, self.encoder, prune=True)
            rows = [normalize(store.get(self.namespace, record.record_id)['vector'])
                    for record in records]
        if rows:
            matrix = np.asarray(rows, dtype=np.float32)
            if matrix.ndim != 2 or not np.isfinite(matrix).all():
                raise ValueError('记忆向量矩阵格式无效')
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            if np.any(norms == 0) or not np.isfinite(norms).all():
                raise ValueError('记忆向量包含零向量或非有限值')
            matrix = np.ascontiguousarray(matrix / norms, dtype=np.float32)
        else:
            matrix = np.empty((0, 0), dtype=np.float32)
        # Keep the session's own snapshot: another session cannot mutate test retrieval.
        self.identities = tuple(record.record_id for record in records)
        self.matrix = matrix
        self.synced = True
        self.stats['syncs'] += 1
        for key in ('encoded_texts', 'backend_calls', 'encoding_ms'):
            self.stats[key] += stats[key]
        self.stats['active_records'] = len(records)
        self.stats['matrix_rows'] = int(matrix.shape[0])
        self.stats['matrix_dimension'] = int(matrix.shape[1]) if matrix.ndim == 2 else 0
        self.stats['matrix_bytes'] = int(matrix.nbytes)
        self.stats['last_sync'] = stats

    def scores(self, dialog, query_vector=None):
        if self.matrix.shape[0] == 0:
            return {}
        if query_vector is not None:
            return self.score_vector(query_vector)
        text = dialog_text(dialog)
        key = hashlib.sha256(text.encode()).hexdigest()
        query = self.query_cache.get(key)
        if query is None:
            start = perf_counter()
            query = self.encoder.encode_texts([text])[0]
            self.stats['encoding_ms'] += (perf_counter() - start) * 1000
            self.stats['backend_calls'] += self.encoder.last_stats['backend_calls']
            self.stats['query_encodings'] += 1
            if len(self.query_cache) >= 256:
                self.query_cache.pop(next(iter(self.query_cache)))
            self.query_cache[key] = query
        else:
            self.stats['query_cache_hits'] += 1
        return self.score_vector(query)

    def score_vector(self, query):
        query_array = np.asarray(normalize(query), dtype=np.float32)
        if query_array.ndim != 1 or self.matrix.shape[1] != query_array.shape[0]:
            raise ValueError('查询向量维度与记忆向量不一致')
        started = perf_counter()
        similarities = np.clip(self.matrix @ query_array, -1.0, 1.0)
        self.stats['searches'] += 1
        self.stats['search_ms'] += (perf_counter() - started) * 1000
        return {identity: float(score)
                for identity, score in zip(self.identities, similarities, strict=True)}
