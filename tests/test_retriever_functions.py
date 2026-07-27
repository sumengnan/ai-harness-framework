# tests/test_retriever_functions.py
from harness.memory.retriever import cosine, mmr_select, recency_score, rrf_fuse


def test_rrf_fuse_combines_ranks():
    scores = rrf_fuse([["a", "b"], ["a", "c"]], rrf_k=60)
    assert scores["a"] > scores["b"]
    assert scores["a"] > scores["c"]
    assert set(scores) == {"a", "b", "c"}


def test_recency_score_decays():
    assert recency_score(0.0, 30.0) == 1.0
    assert recency_score(30.0, 30.0) == 0.5
    assert recency_score(60.0, 30.0) == 0.25
    assert recency_score(-5.0, 30.0) == 1.0
    assert recency_score(10.0, 0.0) == 1.0


def test_cosine():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_mmr_select_reduces_redundancy():
    rel = {"a": 1.0, "b": 0.9, "c": 0.5}
    embs = {"a": [1.0, 0.0], "b": [1.0, 0.0], "c": [0.0, 1.0]}
    picked = mmr_select(["a", "b", "c"], rel, embs, lambda_=0.5, k=2)
    assert picked[0] == "a"
    assert picked[1] == "c"
