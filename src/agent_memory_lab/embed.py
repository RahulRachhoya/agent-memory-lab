"""Embeddings shared by every store, computed once and cached so Qdrant and pgvector index
identical vectors. All models run locally on CPU through fastembed (ONNX); no API calls."""
from pathlib import Path

import numpy as np
from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

DENSE_MODEL = "BAAI/bge-small-en-v1.5"          # 384-d
SPARSE_MODEL = "Qdrant/bm25"                    # term frequencies; Qdrant applies IDF server-side
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
DIM = 384
# bge v1.5 expects this instruction on queries (not on documents) for retrieval.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
CACHE = Path("outputs/cache")

_models: dict[str, object] = {}


def _get(name, cls):
    if name not in _models:
        _models[name] = cls(name)
    return _models[name]


def dense_docs(texts: list[str]) -> np.ndarray:
    path = CACHE / f"dense_{len(texts)}.npy"
    if path.exists():
        return np.load(path)
    vecs = np.array(list(_get(DENSE_MODEL, TextEmbedding).embed(texts, batch_size=64)), dtype=np.float32)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, vecs)
    return vecs


def dense_query(text: str) -> list[float]:
    return next(_get(DENSE_MODEL, TextEmbedding).embed([QUERY_PREFIX + text])).tolist()


def sparse_docs(texts: list[str]):
    return list(_get(SPARSE_MODEL, SparseTextEmbedding).embed(texts, batch_size=256))


def sparse_query(text: str):
    return next(_get(SPARSE_MODEL, SparseTextEmbedding).query_embed(text))


def rerank(query: str, texts: list[str]) -> list[float]:
    return list(_get(RERANK_MODEL, TextCrossEncoder).rerank(query, texts))
