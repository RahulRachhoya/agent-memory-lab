import math

from agent_memory_lab.metrics import ndcg_at_k, percentile, recall_at_k, rrf


def test_ndcg_perfect_and_miss():
    rel = {"a": 1, "b": 1}
    assert ndcg_at_k(["a", "b", "c"], rel) == 1.0
    assert ndcg_at_k(["x", "y"], rel) == 0.0


def test_ndcg_position_discount():
    # One relevant doc at rank 2 instead of rank 1: DCG = 1/log2(3), IDCG = 1.
    assert math.isclose(ndcg_at_k(["x", "a"], {"a": 1}), 1 / math.log2(3))


def test_recall_ignores_zero_relevance():
    assert recall_at_k(["a"], {"a": 1, "b": 1, "c": 0}) == 0.5


def test_rrf_rewards_agreement():
    # "b" is second in both lists, "a" first in one and absent from the other: b wins.
    assert rrf([["a", "b"], ["c", "b"]])[0] == "b"


def test_percentile():
    xs = list(range(1, 101))
    assert percentile(xs, 50) in (50, 51)
    assert percentile(xs, 95) in (95, 96)
