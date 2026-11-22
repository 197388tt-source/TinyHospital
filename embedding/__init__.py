"""Optional semantic text vectors; importing this package does not load a model."""

from .encoder import EmbeddingConfig, Vectorizer
from .records import TextRecord, load_records
from .store import VectorStore

__all__ = ['EmbeddingConfig', 'Vectorizer', 'TextRecord', 'load_records', 'VectorStore']
