"""Qdrant: one collection with a named dense vector and a named BM25 sparse vector, so dense,
sparse and hybrid search all run against the same points. Hybrid fusion happens server-side."""
import time

from qdrant_client import QdrantClient, models

from . import embed

COLLECTION = "scifact"
HNSW_EF = 128   # search-time beam width; must be >= the number of results you want back


def client(grpc: bool = True) -> QdrantClient:
    # gRPC is ~3x faster than REST per query on this setup (~6 ms vs ~18 ms p50 for a dense
    # top-100): most of the REST time is JSON encoding and HTTP, not search.
    return QdrantClient(host="127.0.0.1", port=6333, grpc_port=6334, prefer_grpc=grpc)


def point_id(doc_id: str) -> int:
    # Qdrant ids must be unsigned ints or UUIDs. SciFact ids are integers, so use them directly:
    # id-only searches then return the doc id without reading any payload (payload is on disk).
    return int(doc_id)


def build(qc: QdrantClient, docs, dense, sparse) -> None:
    if qc.collection_exists(COLLECTION):
        qc.delete_collection(COLLECTION)
    qc.create_collection(
        COLLECTION,
        vectors_config={"dense": models.VectorParams(size=embed.DIM, distance=models.Distance.COSINE)},
        # At 5K docs Qdrant would never build HNSW (segments stay under indexing_threshold) and
        # would brute-force every segment (under full_scan_threshold). Both are sensible defaults
        # for a small corpus; lowering them makes the benchmark measure HNSW, as for pgvector.
        hnsw_config=models.HnswConfigDiff(full_scan_threshold=10),
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1),
        # IDF modifier: Qdrant keeps corpus statistics and applies IDF at query time, which
        # together with the fastembed bm25 term weights gives BM25 scoring.
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    points = [
        models.PointStruct(
            id=point_id(d.id),
            vector={"dense": dense[i].tolist(),
                    "bm25": _sv(sparse[i])},
            payload={"title": d.title, "text": d.text},
        )
        for i, d in enumerate(docs)
    ]
    qc.upload_points(COLLECTION, points, batch_size=256, wait=True)
    while True:  # HNSW builds asynchronously after upload
        info = qc.get_collection(COLLECTION)
        if info.status == models.CollectionStatus.GREEN and info.indexed_vectors_count >= 2 * len(docs):  # dense + sparse
            return
        time.sleep(0.5)


def _ids(resp) -> list[str]:
    return [str(p.id) for p in resp.points]


def _sv(s) -> models.SparseVector:
    return models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())


# Search functions take precomputed query vectors (embed.dense_query / embed.sparse_query), so
# timings measure the database, not the embedding model.
def dense(qc: QdrantClient, qvec, limit: int = 100, exact: bool = False) -> list[str]:
    return _ids(qc.query_points(COLLECTION, query=qvec, using="dense", limit=limit,
                                search_params=models.SearchParams(hnsw_ef=HNSW_EF, exact=exact),
                                with_payload=False))


def sparse(qc: QdrantClient, qsparse, limit: int = 100) -> list[str]:
    return _ids(qc.query_points(COLLECTION, query=_sv(qsparse), using="bm25", limit=limit, with_payload=False))


def hybrid(qc: QdrantClient, qvec, qsparse, limit: int = 100, with_text: bool = False, prefetch: int = 100):
    """Dense + BM25 candidates (`prefetch` from each) fused with RRF (k=60) in a single Query API call."""
    resp = qc.query_points(
        COLLECTION,
        prefetch=[
            models.Prefetch(query=qvec, using="dense", limit=max(limit, prefetch),
                            params=models.SearchParams(hnsw_ef=HNSW_EF)),
            models.Prefetch(query=_sv(qsparse), using="bm25", limit=max(limit, prefetch)),
        ],
        query=models.RrfQuery(rrf=models.Rrf(k=60)),
        limit=limit,
        with_payload=with_text,
    )
    if with_text:
        return [(str(p.id), p.payload["title"], p.payload["text"]) for p in resp.points]
    return _ids(resp)


def hybrid_rerank(qc: QdrantClient, text: str, qvec, qsparse, limit: int = 100, top: int = 20) -> list[str]:
    """Hybrid candidates, then a cross-encoder re-scores the top `top`. Recall@100 is unchanged by
    construction; the point is to fix the ordering at the top (nDCG@10). On CPU the cross-encoder
    costs ~70 ms per SciFact abstract, so `top` is the main latency knob."""
    hits = hybrid(qc, qvec, qsparse, limit, with_text=True)
    head, tail = hits[:top], hits[top:]
    scores = embed.rerank(text, [f"{t}\n{b}" for _, t, b in head])
    head = [h for _, h in sorted(zip(scores, head), key=lambda x: x[0], reverse=True)]
    return [d for d, _, _ in head + tail]
