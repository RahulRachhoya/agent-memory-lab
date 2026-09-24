"""Qdrant vs pgvector on BEIR SciFact: same corpus, same cached vectors, same RRF (k=60), same
HNSW beam width (ef=128). Reports nDCG@10, Recall@100 and per-query latency p50/p95.

    uv run python -m agent_memory_lab.bench            # full run (300 test claims)
    uv run python -m agent_memory_lab.bench --no-build # reuse already-built indexes
"""
import argparse
import json
import time
from pathlib import Path

from . import data, embed, metrics, pg_store, qdrant_store

OUT = Path("docs/results.json")
WARMUP = 10


def run_method(fn, queries: dict[str, str], qvecs, qsparse, qrels) -> dict:
    ids = list(queries)
    for qid in ids[:WARMUP]:  # warm caches / connection pools before timing
        fn(queries[qid], qvecs[qid], qsparse[qid])
    ndcg, recall, lat = [], [], []
    for qid in ids:
        t0 = time.perf_counter()
        ranked = fn(queries[qid], qvecs[qid], qsparse[qid])
        lat.append((time.perf_counter() - t0) * 1000)
        ndcg.append(metrics.ndcg_at_k(ranked, qrels[qid], 10))
        recall.append(metrics.recall_at_k(ranked, qrels[qid], 100))
    n = len(ids)
    return {"ndcg@10": sum(ndcg) / n, "recall@100": sum(recall) / n,
            "p50_ms": metrics.percentile(lat, 50), "p95_ms": metrics.percentile(lat, 95)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-build", action="store_true")
    args = ap.parse_args()

    sf = data.load()
    print(f"{len(sf.docs)} docs, {len(sf.queries)} test queries")
    qc, pg = qdrant_store.client(), pg_store.connect()
    pg_planner = pg_store.connect(force_index=False)  # lets Postgres choose; at 5K rows it seq-scans

    if not args.no_build:
        texts = [d.full for d in sf.docs]
        t0 = time.perf_counter()
        dense = embed.dense_docs(texts)
        sparse = embed.sparse_docs(texts)
        print(f"embedded in {time.perf_counter() - t0:.0f}s")
        t0 = time.perf_counter()
        qdrant_store.build(qc, sf.docs, dense, sparse)
        print(f"qdrant built in {time.perf_counter() - t0:.1f}s "
              f"({qc.get_collection(qdrant_store.COLLECTION).indexed_vectors_count} indexed vectors)")
        t0 = time.perf_counter()
        pg_store.build(pg, sf.docs, dense)
        print(f"pgvector built in {time.perf_counter() - t0:.1f}s")

    # Query embeddings are computed once, outside the timed loop.
    qvecs = {qid: embed.dense_query(t) for qid, t in sf.queries.items()}
    qsparse = {qid: embed.sparse_query(t) for qid, t in sf.queries.items()}

    methods = {
        "qdrant dense (exact)": lambda t, v, s: qdrant_store.dense(qc, v, exact=True),
        "qdrant dense": lambda t, v, s: qdrant_store.dense(qc, v),
        "qdrant bm25": lambda t, v, s: qdrant_store.sparse(qc, s),
        "qdrant hybrid (RRF)": lambda t, v, s: qdrant_store.hybrid(qc, v, s),
        "pgvector dense (exact)": lambda t, v, s: pg_store.dense(pg_planner, v),
        "pgvector dense": lambda t, v, s: pg_store.dense(pg, v),
        "postgres FTS": lambda t, v, s: pg_store.fts(pg, t),
        "pgvector hybrid (RRF)": lambda t, v, s: pg_store.hybrid(pg, t, v),
        "qdrant hybrid + rerank": lambda t, v, s: qdrant_store.hybrid_rerank(qc, t, v, s),  # slow: CPU
    }
    results = {}
    for name, fn in methods.items():
        results[name] = run_method(fn, sf.queries, qvecs, qsparse, sf.qrels)
        r = results[name]
        print(f"{name:24s} nDCG@10 {r['ndcg@10']:.3f}  R@100 {r['recall@100']:.3f}  "
              f"p50 {r['p50_ms']:.1f}ms  p95 {r['p95_ms']:.1f}ms", flush=True)

    # Same dense query over Qdrant's two transports, to separate search time from client overhead.
    rest = qdrant_store.client(grpc=False)
    results["qdrant dense (REST)"] = run_method(lambda t, v, s: qdrant_store.dense(rest, v),
                                                sf.queries, qvecs, qsparse, sf.qrels)
    print(f"qdrant dense over REST: p50 {results['qdrant dense (REST)']['p50_ms']:.1f}ms", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"dataset": "BEIR SciFact (test)", "docs": len(sf.docs),
                               "queries": len(sf.queries), "results": results}, indent=2))
    print(f"\n| Method | nDCG@10 | Recall@100 | p50 ms | p95 ms |\n|---|---|---|---|---|")
    for name, r in results.items():
        print(f"| {name} | {r['ndcg@10']:.3f} | {r['recall@100']:.3f} | {r['p50_ms']:.1f} | {r['p95_ms']:.1f} |")


if __name__ == "__main__":
    main()
