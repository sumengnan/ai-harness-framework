from harness.memory.memory import Memory
from harness.memory.sqlite_backend import SqliteVecBackend


async def test_add_texts_chunks_and_stores(mock_embedder):
    backend = SqliteVecBackend(":memory:", dimension=64)
    mem = Memory(backend, mock_embedder(dimension=64), chunk_size=100, overlap=20)
    long_text = "词 " * 200                       # 远超 100 字符 → 多块
    ids = await mem.add_texts([long_text], "knowledge", {"source": "doc1"})
    assert len(ids) >= 2                           # 分成多块入库


async def test_search_recalls_relevant(mock_embedder):
    backend = SqliteVecBackend(":memory:", dimension=64)
    mem = Memory(backend, mock_embedder(dimension=64), chunk_size=1000, overlap=0)
    await mem.add_texts(["python programming language"], "knowledge")
    await mem.add_texts(["the cat sat on the mat"], "knowledge")
    hits = await mem.search("cat", "knowledge", k=1)
    assert hits[0].text == "the cat sat on the mat"   # 共享 "cat" → 更近


async def test_add_empty_text_returns_empty(mock_embedder):
    backend = SqliteVecBackend(":memory:", dimension=64)
    mem = Memory(backend, mock_embedder(dimension=64), chunk_size=100, overlap=20)
    assert await mem.add_texts(["   "], "knowledge") == []
