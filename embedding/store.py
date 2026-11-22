"""Transactional, separate vector cache. Original datasets/memories are read-only.

One database belongs to one model/preprocessing configuration. Namespaces keep
train/test/dialog/experience sets separate; synchronization never crosses them.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import struct
from contextlib import closing
from pathlib import Path
from time import perf_counter

from .encoder import EmbeddingConfig, Vectorizer, normalize
from .records import TextRecord, clean_text


class VectorStore:
    def __init__(self, path: Path, config: EmbeddingConfig):
        self.path, self.config = Path(path), config
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=30)
        self.db.execute('PRAGMA foreign_keys = ON')
        try:
            with self.db:
                self.db.executescript('''
                    CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS vectors (
                        text_hash TEXT PRIMARY KEY, dimension INTEGER NOT NULL, value BLOB NOT NULL);
                    CREATE TABLE IF NOT EXISTS records (
                        namespace TEXT NOT NULL, record_id TEXT NOT NULL,
                        text TEXT NOT NULL, kind TEXT NOT NULL, source TEXT NOT NULL,
                        text_hash TEXT NOT NULL REFERENCES vectors(text_hash),
                        PRIMARY KEY (namespace, record_id));
                ''')
                self.db.execute('BEGIN IMMEDIATE')
                row = self.db.execute("SELECT value FROM meta WHERE key='fingerprint'").fetchone()
                if row and row[0] != config.fingerprint:
                    raise ValueError('向量库模型或预处理配置不同；请使用新的 --output，禁止混用向量')
                self.db.execute('INSERT OR IGNORE INTO meta VALUES (?, ?)', ('fingerprint', config.fingerprint))
                self.db.execute('INSERT OR IGNORE INTO meta VALUES (?, ?)',
                                ('manifest', json.dumps(config.manifest, ensure_ascii=False, sort_keys=True)))
        except Exception:
            self.db.close()
            raise

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def sync(self, namespace: str, records: list[TextRecord], encoder: Vectorizer,
             prune: bool = False) -> dict:
        """Upsert supplied records; remove absent IDs only with explicit prune.

        Encoding failures roll back this whole import. Removed cache vectors may
        remain to support repeat experiments; they are not active search records.
        """
        if not namespace.strip():
            raise ValueError('namespace 不能为空')
        if encoder.config.fingerprint != self.config.fingerprint:
            raise ValueError('编码器与向量库配置不匹配')
        prepared, ids = [], set()
        for record in records:
            if not isinstance(record.record_id, str) or not record.record_id.strip() or record.record_id in ids:
                raise ValueError('记录 ID 为空或重复')
            ids.add(record.record_id)
            text = clean_text(record.text)
            digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
            prepared.append((record, text, digest))
        started = perf_counter()
        stats = {'namespace': namespace, 'records': len(prepared), 'added': 0,
                 'updated': 0, 'unchanged': 0, 'deleted': 0, 'encoded_texts': 0,
                 'reused_records': 0, 'backend_calls': 0, 'chunks': 0,
                 'input_characters': 0, 'encoding_ms': 0.0}
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            previous = dict(self.db.execute('SELECT record_id, text_hash FROM records WHERE namespace=?', (namespace,)))
            missing = {}
            for record, text, digest in prepared:
                category = ('added' if record.record_id not in previous else
                            'unchanged' if previous[record.record_id] == digest else 'updated')
                stats[category] += 1
                if not self.db.execute('SELECT 1 FROM vectors WHERE text_hash=?', (digest,)).fetchone():
                    missing[digest] = text
            items = list(missing.items())
            row = self.db.execute("SELECT value FROM meta WHERE key='dimension'").fetchone()
            dimension = int(row[0]) if row else None
            for start in range(0, len(items), encoder.batch_size):
                batch = items[start:start + encoder.batch_size]
                before = perf_counter()
                vectors = encoder.encode_texts([text for _, text in batch])
                stats['encoding_ms'] += (perf_counter() - before) * 1000
                for key in ('backend_calls', 'chunks', 'input_characters'):
                    stats[key] += encoder.last_stats[key]
                for (digest, _), vector in zip(batch, vectors):
                    if dimension is None:
                        dimension = len(vector)
                        self.db.execute('INSERT INTO meta VALUES (?, ?)', ('dimension', str(dimension)))
                    if len(vector) != dimension:
                        raise ValueError('模型向量维度发生变化，已取消本次导入')
                    blob = struct.pack(f'<{dimension}f', *vector)
                    self.db.execute('INSERT INTO vectors VALUES (?, ?, ?)', (digest, dimension, blob))
            for record, text, digest in prepared:
                self.db.execute('''INSERT INTO records VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(namespace, record_id) DO UPDATE SET
                    text=excluded.text, kind=excluded.kind, source=excluded.source,
                    text_hash=excluded.text_hash''',
                    (namespace, record.record_id, text, record.kind, record.source, digest))
            if prune:
                removed = previous.keys() - ids
                self.db.executemany('DELETE FROM records WHERE namespace=? AND record_id=?',
                                    [(namespace, identity) for identity in removed])
                stats['deleted'] = len(removed)
            stats['encoded_texts'] = len(items)
            stats['reused_records'] = len(prepared) - len(items)
            stats['dimension'] = dimension
            stats['new_vector_bytes'] = len(items) * (dimension or 0) * 4
            stats['active_records'] = self.db.execute('SELECT count(*) FROM records WHERE namespace=?', (namespace,)).fetchone()[0]
            stats['total_ms'] = (perf_counter() - started) * 1000
            self.db.execute('INSERT OR REPLACE INTO meta VALUES (?, ?)',
                            ('last_sync', json.dumps(stats, ensure_ascii=False)))
        return stats

    def get(self, namespace: str, record_id: str) -> dict:
        row = self.db.execute('''SELECT r.text, r.kind, r.source, r.text_hash, v.dimension, v.value
            FROM records r JOIN vectors v USING(text_hash)
            WHERE r.namespace=? AND r.record_id=?''', (namespace, record_id)).fetchone()
        if row is None:
            raise KeyError((namespace, record_id))
        text, kind, source, digest, dimension, blob = row
        return {'record_id': record_id, 'namespace': namespace, 'text': text,
                'kind': kind, 'source': source, 'text_hash': digest,
                'vector': list(struct.unpack(f'<{dimension}f', blob))}

    def search(self, namespace: str, query: list[float], limit: int = 3,
               min_similarity: float = 0.0, exclude_ids=()) -> list[dict]:
        """Exact cosine Top-K over active records in one namespace."""
        if type(limit) is not int or limit < 0:
            raise ValueError('limit 必须为非负整数')
        if not -1 <= min_similarity <= 1:
            raise ValueError('相似度阈值必须在 -1 到 1 之间')
        query = normalize(query)
        dimension = self.db.execute("SELECT value FROM meta WHERE key='dimension'").fetchone()
        if dimension and len(query) != int(dimension[0]):
            raise ValueError('查询向量维度与向量库不一致')
        excluded, ranked = set(exclude_ids), []
        for (identity,) in self.db.execute(
                'SELECT record_id FROM records WHERE namespace=?', (namespace,)):
            if identity in excluded:
                continue
            row = self.get(namespace, identity)
            score = sum(a * b for a, b in zip(query, normalize(row.pop('vector'))))
            score = max(-1.0, min(1.0, score))
            if score >= min_similarity:
                ranked.append({**row, 'similarity': score})
        ranked.sort(key=lambda row: (-row['similarity'], row['record_id']))
        return ranked[:limit]


def inspect_store(path: Path) -> dict:
    """Read existing output without loading a model or creating an empty DB."""
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)) as db:
        metadata = dict(db.execute('SELECT key, value FROM meta'))
        return {'manifest': json.loads(metadata['manifest']),
                'dimension': int(metadata['dimension']) if 'dimension' in metadata else None,
                'namespaces': dict(db.execute('SELECT namespace, count(*) FROM records GROUP BY namespace')),
                'cached_vectors': db.execute('SELECT count(*) FROM vectors').fetchone()[0],
                'last_sync': json.loads(metadata.get('last_sync', '{}'))}
