# tests/test_retrieval_eval.py
from harness.memory.eval import GoldenCase, evaluate
from harness.memory.record import MemoryFilter
from harness.memory.reranker import NoOpReranker
from harness.memory.retriever import RetrievalConfig, Retriever
from harness.memory.sqlite_backend import SqliteVecBackend

_CORPUS = [
    ("快速排序是最常见的排序算法之一", "d1"),
    ("二叉树是一种基础数据结构", "d2"),
    ("图的广度优先遍历用于最短路径", "d3"),
    ("动态规划用于求解最优子结构问题", "d4"),
]
_CASES = [
    GoldenCase(query="排序算法", relevant_ids={"d1"}),
    GoldenCase(query="数据结构", relevant_ids={"d2"}),
    GoldenCase(query="广度优先", relevant_ids={"d3"}),
    GoldenCase(query="动态规划", relevant_ids={"d4"}),
]


async def _seed(mock_embedder):
    from harness.memory.record import MemType, MemoryRecord
    backend = SqliteVecBackend(":memory:", dimension=64)
    emb = mock_embedder(dimension=64)
    for text, rid in _CORPUS:
        v = (await emb.embed([text]))[0]
        backend.upsert([MemoryRecord(owner_id="u1", kind="k",
                                     mem_type=MemType.SEMANTIC, text=text,
                                     embedding=v, id=rid)])
    return backend, emb


def _retriever(backend, emb, cfg):
    return Retriever(backend, emb, NoOpReranker(), cfg)


async def test_metrics_computed(mock_embedder):
    backend, emb = await _seed(mock_embedder)
    cfg = RetrievalConfig()
    m = await evaluate(_retriever(backend, emb, cfg), MemoryFilter(owner_id="u1"), _CASES, k=3)
    assert 0.0 <= m["hit@k"] <= 1.0 and 0.0 <= m["mrr"] <= 1.0


async def test_hybrid_beats_vector_only(mock_embedder):
    backend, emb = await _seed(mock_embedder)
    vector_only = RetrievalConfig(use_keyword=False, use_mmr=False,
                                  w_recency=0.0, w_importance=0.0)
    hybrid = RetrievalConfig(use_keyword=True, use_mmr=False,
                             w_recency=0.0, w_importance=0.0)
    flt = MemoryFilter(owner_id="u1")
    mv = await evaluate(_retriever(backend, emb, vector_only), flt, _CASES, k=3)
    mh = await evaluate(_retriever(backend, emb, hybrid), flt, _CASES, k=3)
    assert mh["hit@k"] >= mv["hit@k"]
    assert mh["mrr"] >= mv["mrr"]


from harness.memory.eval import store_size
from harness.memory.maintainer import ConsolidationConfig, MemoryMaintainer


class _Scripted:
    def __init__(self, responses):
        self._r = list(responses)
    async def __call__(self, s, u):
        return self._r.pop(0)


async def test_consolidation_preserves_recall_reduces_size(mock_embedder):
    from harness.memory.record import MemType, MemoryFilter, MemoryRecord
    from harness.memory.reranker import NoOpReranker
    from harness.memory.retriever import RetrievalConfig, Retriever
    from harness.memory.sqlite_backend import SqliteVecBackend
    backend = SqliteVecBackend(":memory:", dimension=64)
    emb = mock_embedder(dimension=64)
    distilled = "用户偏好深色主题"
    v = (await emb.embed(["深色 主题 偏好"]))[0]
    for rid in ("a", "b", "c"):
        backend.upsert([MemoryRecord(owner_id="u1", kind="k", mem_type=MemType.EPISODIC,
                                     text=f"深色 主题 偏好 {rid}", embedding=v, id=rid)])
    size_before = store_size(backend, "u1", "k")
    assert size_before == 3
    m = MemoryMaintainer(backend, emb, _Scripted([distilled]),
                         ConsolidationConfig(min_cluster=2), now_fn=lambda: 1000)
    await m.consolidate("u1", "k")
    size_after = store_size(backend, "u1", "k")
    assert size_after < size_before
    retr = Retriever(backend, emb, NoOpReranker(), RetrievalConfig())
    hits = await retr.retrieve(distilled, MemoryFilter(owner_id="u1", kind="k"), k=5)
    assert any(h.record.text == distilled for h in hits)
