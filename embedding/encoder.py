"""Lazy real embedding backends and deterministic, versioned preprocessing.

No hash/random fallback is offered: those are not semantic embeddings.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from tool.api_retry import request_json

from .records import PREPROCESS_VERSION, clean_text


@dataclass(frozen=True)
class EmbeddingConfig:
    backend: str
    model: str
    model_version: str
    chunk_chars: int = 256
    prefix: str = ''
    endpoint: str = ''
    revision: str = ''
    device: str = 'cpu'
    local_files_only: bool = True

    def __post_init__(self):
        if self.backend not in ('local', 'api'):
            raise ValueError('backend 必须为 local 或 api')
        if not self.model.strip() or not self.model_version.strip():
            raise ValueError('必须指定模型名称和不可混用的模型版本标记')
        if type(self.chunk_chars) is not int or self.chunk_chars < 1:
            raise ValueError('chunk_chars 必须是正整数')
        if self.backend == 'api':
            url = urlsplit(self.endpoint)
            if url.scheme not in ('http', 'https') or not url.netloc or url.username or url.password or url.query or url.fragment:
                raise ValueError('endpoint 必须是完整 embeddings URL，不能包含凭据、查询或片段')

    @property
    def manifest(self) -> dict:
        return {**asdict(self), 'preprocessing': PREPROCESS_VERSION,
                'aggregation': 'normalized-chunks-character-weighted-mean-v1',
                'normalization': 'l2', 'storage': 'little-endian-float32', 'schema': 1}

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.manifest, sort_keys=True,
                                        ensure_ascii=False).encode()).hexdigest()


def normalize(vector) -> list[float]:
    if not isinstance(vector, (list, tuple)) or not vector:
        raise ValueError('模型必须返回非空数值向量')
    if any(type(x) not in (float, int) or not math.isfinite(x) for x in vector):
        raise ValueError('向量包含非数值或 NaN/Inf')
    scale = max(abs(x) for x in vector)
    if scale == 0:
        raise ValueError('模型返回零向量')
    scaled = [x / scale for x in vector]
    norm = math.sqrt(sum(x * x for x in scaled))
    return [x / norm for x in scaled]


class ApiBackend:
    """JSON embeddings endpoint accepting model/input/encoding_format.

    The complete endpoint is explicit; chat credentials are never reused.
    """
    def __init__(self, config: EmbeddingConfig, api_key_env: str, timeout: float):
        self.config, self.api_key_env, self.timeout = config, api_key_env, timeout

    def encode(self, texts: list[str]) -> list[list[float]]:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise ValueError(f'请设置环境变量 {self.api_key_env}')
        request = Request(self.config.endpoint, method='POST', headers={
            'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'},
            data=json.dumps({'model': self.config.model, 'input': texts,
                             'encoding_format': 'float'}, ensure_ascii=False).encode())
        try:
            value = request_json(urlopen, request, self.timeout)
        except HTTPError as exc:
            raise RuntimeError(f'embedding API HTTP {exc.code}；检查地址、模型和授权') from None
        except URLError:
            raise RuntimeError('embedding API 连接失败；检查网络和地址') from None
        data = value.get('data') if isinstance(value, dict) else None
        if not isinstance(data, list) or len(data) != len(texts):
            raise ValueError('embedding API 返回数量与输入不一致')
        indexed = {}
        for row in data:
            if not isinstance(row, dict) or type(row.get('index')) is not int:
                raise ValueError('embedding API 缺少有效 index')
            index = row['index']
            if index in indexed or not 0 <= index < len(texts):
                raise ValueError('embedding API index 重复或越界')
            indexed[index] = row.get('embedding')
        return [indexed[i] for i in range(len(texts))]


class LocalBackend:
    def __init__(self, config: EmbeddingConfig):
        self.config, self.model = config, None

    def encode(self, texts: list[str]) -> list[list[float]]:
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise RuntimeError('本地模型依赖未安装：python -m pip install -r embedding/requirements-local.txt') from None
            self.model = SentenceTransformer(
                self.config.model, revision=self.config.revision or None,
                device=self.config.device, local_files_only=self.config.local_files_only,
                trust_remote_code=False)
        # Do not silently truncate long records or model-specific prefixes.
        tokenized = self.model.tokenizer(texts, truncation=False, padding=False)
        if any(len(ids) > self.model.max_seq_length for ids in tokenized['input_ids']):
            raise ValueError('分段超过模型 token 上限；减小 --chunk-chars，并使用新的向量库')
        return self.model.encode(texts, batch_size=len(texts), prompt='',
                                 normalize_embeddings=False, convert_to_numpy=True,
                                 show_progress_bar=False).tolist()


class Vectorizer:
    def __init__(self, config: EmbeddingConfig, batch_size: int = 32,
                 api_key_env: str = 'EMBEDDING_API_KEY', timeout: float = 120,
                 backend=None):
        if type(batch_size) is not int or batch_size < 1 or timeout <= 0:
            raise ValueError('batch_size 和 timeout 必须大于零')
        self.config, self.batch_size = config, batch_size
        self.backend = backend if backend is not None else (
            LocalBackend(config) if config.backend == 'local'
            else ApiBackend(config, api_key_env, timeout))
        self.last_stats = {}

    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        cleaned = [clean_text(text) for text in texts]
        jobs = [(i, text[start:start + self.config.chunk_chars])
                for i, text in enumerate(cleaned)
                for start in range(0, len(text), self.config.chunk_chars)]
        sums, dimension = [None] * len(texts), None
        calls = 0
        for start in range(0, len(jobs), self.batch_size):
            batch = jobs[start:start + self.batch_size]
            encoded = self.backend.encode([self.config.prefix + chunk for _, chunk in batch])
            calls += 1
            if len(encoded) != len(batch):
                raise ValueError('模型返回向量数量不匹配')
            for (index, chunk), raw in zip(batch, encoded):
                vector = normalize(raw)
                if dimension is None:
                    dimension = len(vector)
                if len(vector) != dimension:
                    raise ValueError('同一模型返回了不同维度')
                if sums[index] is None:
                    sums[index] = [0.0] * dimension
                for j, value in enumerate(vector):
                    sums[index][j] += value * len(chunk)
        result = [normalize(vector) for vector in sums]
        self.last_stats = {'backend_calls': calls, 'chunks': len(jobs),
                           'input_characters': sum(len(x) for x in cleaned)}
        return result
