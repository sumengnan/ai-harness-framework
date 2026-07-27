# tests/test_memory_retrieve.py
from harness.memory.memory import Memory
from harness.memory.retriever import ScoredHit
from harness.memory.sqlite_backend import SqliteVecBackend


def _mem(mock_embedder):
    return Memory(SqliteVecBackend(":memory:", dimension=64),
                  mock_embedder(dimension=64), chunk_size=1000, overlap=0)


async def test_search_still_returns_old_hit_shape(mock_embedder):
    m = _mem(mock_embedder)
    await m.add_texts(["the cat sat on the mat"], "knowledge:u1")
    await m.add_texts(["python programming language"], "knowledge:u1")
    hits = await m.search("cat", "knowledge:u1", k=1)
    assert hits[0].text == "the cat sat on the mat"
    assert hits[0].collection == "knowledge:u1"
    assert hasattr(hits[0], "distance") and hasattr(hits[0], "metadata")


async def test_retrieve_exposes_scored_hits(mock_embedder):
    m = _mem(mock_embedder)
    await m.add_texts(["二叉树是一种数据结构"], "knowledge:u1")
    hits = await m.retrieve("数据结构", "knowledge:u1", k=1)
    assert isinstance(hits[0], ScoredHit)
    assert "relevance" in hits[0].components


async def test_hybrid_recalls_chinese_substring(mock_embedder):
    # mock 向量对无空格中文近乎无效，关键词侧（trigram）应把它召回 → hybrid 生效
    m = _mem(mock_embedder)
    await m.add_texts(["快速排序是最常见的排序算法之一"], "knowledge:u1")
    await m.add_texts(["图的广度优先遍历"], "knowledge:u1")
    hits = await m.search("排序算法", "knowledge:u1", k=1)
    assert hits[0].text == "快速排序是最常见的排序算法之一"
