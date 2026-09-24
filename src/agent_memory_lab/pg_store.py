"""Postgres + pgvector: HNSW over the same dense vectors, plus built-in full-text search
(tsvector / ts_rank_cd) as the lexical side. Postgres FTS is not BM25: there is no IDF and no
length normalisation by default, which the benchmark makes visible."""
import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from . import embed

DSN = "postgresql://lab:lab@localhost:5432/lab"
EF_SEARCH = 128  # pgvector's default is 40, which silently caps an HNSW query at ~40 rows


def connect(dsn: str = DSN, force_index: bool = True) -> psycopg.Connection:
    conn = psycopg.connect(dsn, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    conn.execute(f"SET hnsw.ef_search = {EF_SEARCH}")
    # At 5K rows the planner prefers a seq scan + top-N sort (exact search) over HNSW. Disabling
    # seq scans forces the HNSW and GIN indexes, so the benchmark measures the indexes.
    if force_index:
        conn.execute("SET enable_seqscan = off")
    return conn


def build(conn: psycopg.Connection, docs, dense) -> None:
    conn.execute("DROP TABLE IF EXISTS docs")
    conn.execute(f"""
        CREATE TABLE docs (
            id text PRIMARY KEY,
            title text NOT NULL,
            body text NOT NULL,
            embedding vector({embed.DIM}) NOT NULL,
            tsv tsvector GENERATED ALWAYS AS (
                setweight(to_tsvector('english', title), 'A') || to_tsvector('english', body)) STORED
        )""")
    with conn.cursor() as cur, cur.copy("COPY docs (id, title, body, embedding) FROM STDIN WITH (FORMAT BINARY)") as cp:
        cp.set_types(["text", "text", "text", "vector"])
        for d, v in zip(docs, dense):
            cp.write_row([d.id, d.title, d.text, np.asarray(v, dtype=np.float32)])
    conn.execute("CREATE INDEX ON docs USING hnsw (embedding vector_cosine_ops)")
    conn.execute("CREATE INDEX ON docs USING gin (tsv)")
    conn.execute("ANALYZE docs")


# plainto_tsquery ANDs every term, so a long claim matches almost nothing. Swapping & for |
# turns it into an OR query, which is what a ranked keyword search needs.
_TSQUERY = "to_tsquery('english', replace(plainto_tsquery('english', %(q)s)::text, ' & ', ' | '))"


# Dense search takes a precomputed query vector (embed.dense_query) so timings measure the database.
def dense(conn, qvec, limit: int = 100) -> list[str]:
    rows = conn.execute("SELECT id FROM docs ORDER BY embedding <=> %(v)s LIMIT %(n)s",
                        {"v": np.asarray(qvec, dtype=np.float32), "n": limit}).fetchall()
    return [r[0] for r in rows]


def fts(conn, text: str, limit: int = 100) -> list[str]:
    rows = conn.execute(f"""SELECT id FROM docs, {_TSQUERY} q WHERE tsv @@ q
                            ORDER BY ts_rank_cd(tsv, q) DESC LIMIT %(n)s""", {"q": text, "n": limit}).fetchall()
    return [r[0] for r in rows]


def hybrid(conn, text: str, qvec, limit: int = 100) -> list[str]:
    """Dense + FTS fused with RRF (k=60) inside one SQL statement."""
    rows = conn.execute(f"""
        WITH d AS (
            SELECT id, row_number() OVER (ORDER BY embedding <=> %(v)s) AS r
            FROM docs ORDER BY embedding <=> %(v)s LIMIT %(n)s
        ), l AS (
            SELECT id, row_number() OVER (ORDER BY ts_rank_cd(tsv, q) DESC) AS r
            FROM docs, {_TSQUERY} q WHERE tsv @@ q ORDER BY ts_rank_cd(tsv, q) DESC LIMIT %(n)s
        )
        SELECT id FROM (
            SELECT id, 1.0 / (60 + r) AS s FROM d UNION ALL SELECT id, 1.0 / (60 + r) FROM l
        ) u GROUP BY id ORDER BY sum(s) DESC LIMIT %(n)s""",
        {"v": np.asarray(qvec, dtype=np.float32), "q": text, "n": limit}).fetchall()
    return [r[0] for r in rows]
