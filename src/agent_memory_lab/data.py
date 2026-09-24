"""BEIR SciFact: 5,183 abstracts, 300 test claims with graded relevance judgements (qrels)."""
import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from dataclasses import dataclass

from datasets import load_dataset


@dataclass
class Doc:
    id: str
    title: str
    text: str

    @property
    def full(self) -> str:
        return f"{self.title}\n{self.text}"


@dataclass
class SciFact:
    docs: list[Doc]
    queries: dict[str, str]            # query id -> text (test split only)
    qrels: dict[str, dict[str, int]]   # query id -> {doc id: relevance}


def load() -> SciFact:
    corpus = load_dataset("mteb/scifact", "corpus", split="corpus")
    queries = load_dataset("mteb/scifact", "queries", split="queries")
    judged = load_dataset("mteb/scifact", "default", split="test")

    qrels: dict[str, dict[str, int]] = {}
    for r in judged:
        qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])
    text = {str(q["_id"]): q["text"] for q in queries}
    return SciFact(
        docs=[Doc(str(d["_id"]), d["title"], d["text"]) for d in corpus],
        queries={qid: text[qid] for qid in sorted(qrels, key=int)},
        qrels=qrels,
    )
