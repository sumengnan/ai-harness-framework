# tests/test_backend_keyword.py
from harness.memory.record import MemType, MemoryFilter, MemoryRecord
from harness.memory.sqlite_backend import SqliteVecBackend


def _rec(owner, kind, text, vec, mem_type=MemType.SEMANTIC, rid=None):
    r = MemoryRecord(owner_id=owner, kind=kind, mem_type=mem_type, text=text, embedding=vec)
    if rid:
        r.id = rid
    return r


def _b():
    return SqliteVecBackend(":memory:", dimension=3)


def test_keyword_cjk_substring():
    b = _b()
    b.upsert([_rec("u1", "k", "二叉树是一种数据结构", [1.0, 0.0, 0.0]),
              _rec("u1", "k", "快速排序是一种排序算法", [0.0, 1.0, 0.0])])
    hits = b.keyword_search("数据结构", filters=MemoryFilter(owner_id="u1"), k=5)
    assert [h.record.text for h in hits] == ["二叉树是一种数据结构"]


def test_keyword_pushdown_filters():
    b = _b()
    b.upsert([_rec("u1", "knowledge", "排序算法讲解", [1.0, 0.0, 0.0]),
              _rec("u1", "conversation", "排序算法讨论", [0.0, 1.0, 0.0]),
              _rec("u2", "knowledge", "排序算法笔记", [0.0, 0.0, 1.0])])
    hits = b.keyword_search("排序算法",
                            filters=MemoryFilter(owner_id="u1", kind="knowledge"), k=5)
    assert [h.record.text for h in hits] == ["排序算法讲解"]


def test_keyword_excludes_superseded():
    b = _b()
    r = _rec("u1", "k", "已废弃的排序算法", [1.0, 0.0, 0.0], rid="s1")
    r.superseded = 1
    b.upsert([r])
    assert b.keyword_search("排序算法", filters=MemoryFilter(owner_id="u1"), k=5) == []


def test_keyword_short_or_no_match_empty():
    b = _b()
    b.upsert([_rec("u1", "k", "二叉树", [1.0, 0.0, 0.0])])
    assert b.keyword_search("树", filters=MemoryFilter(owner_id="u1"), k=5) == []
    assert b.keyword_search("红黑树", filters=MemoryFilter(owner_id="u1"), k=5) == []


def test_keyword_deleted_not_matched():
    b = _b()
    b.upsert([_rec("u1", "k", "归并排序算法", [1.0, 0.0, 0.0], rid="d1")])
    b.delete(["d1"])
    assert b.keyword_search("归并排序", filters=MemoryFilter(owner_id="u1"), k=5) == []


def test_get_embeddings_by_id():
    b = _b()
    b.upsert([_rec("u1", "k", "x", [1.0, 0.0, 0.0], rid="a"),
              _rec("u1", "k", "y", [0.0, 1.0, 0.0], rid="c")])
    embs = b.get_embeddings(["a", "c", "missing"])
    assert set(embs.keys()) == {"a", "c"}
    assert embs["a"] == [1.0, 0.0, 0.0]


def test_keyword_punctuation_queries():
    b = _b()
    b.upsert([_rec("u1", "k", "C++ 是一门系统编程语言", [1.0, 0.0, 0.0]),
              _rec("u1", "k", "foo-bar 是占位命名", [0.0, 1.0, 0.0]),
              _rec("u1", "k", "冒号语法 a:b 的示例", [0.0, 0.0, 1.0])])
    assert [h.record.text for h in b.keyword_search("C++", filters=MemoryFilter(owner_id="u1"), k=5)] == ["C++ 是一门系统编程语言"]
    assert [h.record.text for h in b.keyword_search("foo-bar", filters=MemoryFilter(owner_id="u1"), k=5)] == ["foo-bar 是占位命名"]
    assert [h.record.text for h in b.keyword_search("a:b", filters=MemoryFilter(owner_id="u1"), k=5)] == ["冒号语法 a:b 的示例"]


def test_keyword_cjk_still_works_after_escape():
    b = _b()
    b.upsert([_rec("u1", "k", "快速排序算法讲解", [1.0, 0.0, 0.0])])
    assert [h.record.text for h in b.keyword_search("排序算法", filters=MemoryFilter(owner_id="u1"), k=5)] == ["快速排序算法讲解"]
