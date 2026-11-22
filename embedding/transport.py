"""Typed float32 semantic sidecars; never JSON arrays or chat prompt numbers."""
from hashlib import sha256
import math
import struct
from time import perf_counter

from protocol import validate_embedding_descriptor
from shared_transport import SharedPayloadArena, read_reference, validate_reference
from .encoder import EmbeddingConfig, Vectorizer, normalize
from .retrieval import dialog_text


def make_vectorizer(options=None):
    options = dict(options or {})
    config = options.get('config') or EmbeddingConfig(
        'api', 'Qwen3-Embedding-8B', 'Qwen3-Embedding-8B',
        endpoint='http://aigw.dlut.edu.cn/v1/embeddings')
    if isinstance(config, dict):
        config = EmbeddingConfig(**config)
    return Vectorizer(config, batch_size=options.get('batch_size', 32),
                      api_key_env=options.get('api_key_env', 'EMBEDDING_API_KEY'),
                      timeout=options.get('timeout', 120))


class QueryVectorSender:
    def __init__(self, mode, options=None):
        if mode not in ('shared_memory', 'binary'):
            raise ValueError('向量发送模式无效')
        self.mode, self.encoder = mode, make_vectorizer(options)
        self.arena = SharedPayloadArena()
        self.cache = {}
        self.stats = dict(query_encodings=0, cache_hits=0, backend_calls=0,
                          encoding_ms=0.0, shared_write_bytes=0, binary_bytes=0)

    def prepare(self, dialog, trace_id, name=None):
        text = dialog_text(dialog)
        text_hash = sha256(text.encode()).hexdigest()
        vector = self.cache.get(text_hash)
        if vector is None:
            started = perf_counter()
            vector = self.encoder.encode_texts([text])[0]
            self.stats['encoding_ms'] += (perf_counter() - started) * 1000
            self.stats['query_encodings'] += 1
            self.stats['backend_calls'] += self.encoder.last_stats['backend_calls']
            if len(self.cache) >= 256:
                self.cache.pop(next(iter(self.cache)))
            self.cache[text_hash] = vector
        else:
            self.stats['cache_hits'] += 1
        raw = struct.pack(f'<{len(vector)}f', *vector)
        descriptor = dict(version=1, kind='embedding', storage=self.mode,
                          model_fingerprint=self.encoder.config.fingerprint,
                          dimension=len(vector), dtype='float32-le', normalized=True,
                          size=len(raw), sha256=sha256(raw).hexdigest(),
                          text_hash=text_hash, trace_id=trace_id)
        if self.mode == 'shared_memory':
            ref, _, written = self.arena.publish(raw, trace_id, name=name)
            descriptor['reference'] = ref
            self.stats['shared_write_bytes'] += written
        else:
            self.stats['binary_bytes'] += len(raw)
        validate_embedding_descriptor(descriptor, trace_id)
        return descriptor, raw if self.mode == 'binary' else None

    def release(self):
        self.arena.release()


def receive_vector(descriptor, *, trace_id, fingerprint, dialog=None, binary=None):
    validate_embedding_descriptor(descriptor, trace_id)
    if descriptor['model_fingerprint'] != fingerprint:
        raise ValueError('查询向量与记忆索引模型或预处理配置不一致')
    # In vector-only IPC mode Retrieval deliberately never receives the source
    # dialogue.  The parent Runtime has already authenticated the sender and
    # trace; content integrity is still checked through the vector SHA-256.
    if dialog is not None and descriptor['text_hash'] != sha256(dialog_text(dialog).encode()).hexdigest():
        raise ValueError('查询向量与当前对话不匹配')
    if descriptor['storage'] == 'shared_memory':
        if binary is not None:
            raise ValueError('共享向量不应附带二进制载荷')
        ref = descriptor['reference']
        validate_reference(ref, trace_id)
        if ref['size'] != descriptor['size'] or ref['sha256'] != descriptor['sha256']:
            raise ValueError('共享向量引用与描述不一致')
        raw, _ = read_reference(ref, trace_id)
    else:
        raw = binary
    if (not isinstance(raw, bytes) or len(raw) != descriptor['size']
            or sha256(raw).hexdigest() != descriptor['sha256']):
        raise ValueError('向量载荷缺失、长度不符或校验失败')
    vector = list(struct.unpack(f"<{descriptor['dimension']}f", raw))
    if any(not math.isfinite(x) for x in vector) or abs(sum(x*x for x in vector) - 1) > 1e-4:
        raise ValueError('向量必须为有限的单位向量')
    return normalize(vector)
