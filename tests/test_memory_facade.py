from harness.memory.memory import Memory, collection_to_scope
from harness.memory.sqlite_backend import SqliteVecBackend


def test_collection_to_scope_mapping():
    assert collection_to_scope("knowledge:u1") == ("u1", "knowledge")
    assert collection_to_scope("conversation:c1") == ("c1", "conversation")
    assert collection_to_scope("knowledge") == ("_global", "knowledge")
    assert collection_to_scope("episodes") == ("_global", "episodes")


async def test_add_and_search_roundtrip(mock_embedder):
    backend = SqliteVecBackend(":memory:", dimension=64)
    mem = Memory(backend, mock_embedder(dimension=64), chunk_size=1000, overlap=0)
    await mem.add_texts(["python programming language"], "knowledge:u1")
    await mem.add_texts(["the cat sat on the mat"], "knowledge:u1")
    hits = await mem.search("cat", "knowledge:u1", k=1)
    assert hits[0].text == "the cat sat on the mat"
    assert hits[0].collection == "knowledge:u1"     # 回填原 collection 字符串


async def test_search_isolated_by_owner(mock_embedder):
    backend = SqliteVecBackend(":memory:", dimension=64)
    mem = Memory(backend, mock_embedder(dimension=64), chunk_size=1000, overlap=0)
    await mem.add_texts(["the cat sat on the mat"], "knowledge:u1")
    assert await mem.search("cat", "knowledge:u2", k=5) == []   # 不串租户
