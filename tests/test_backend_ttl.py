# tests/test_backend_ttl.py
from harness.memory.record import MemType, MemoryFilter, MemoryRecord
from harness.memory.sqlite_backend import SqliteVecBackend


def _rec(rid, expires_at, vec=(1.0, 0.0, 0.0)):
    r = MemoryRecord(owner_id="u1", kind="k", mem_type=MemType.SEMANTIC,
                     text=f"m{rid}", embedding=list(vec), id=rid)
    r.expires_at = expires_at
    return r


def _b():
    return SqliteVecBackend(":memory:", dimension=3, now_fn=lambda: 1000)


def test_vector_search_excludes_expired():
    b = _b()
    b.upsert([_rec("never", 0), _rec("expired", 100), _rec("future", 5000)])
    ids = [h.record.id for h in b.vector_search(
        [1.0, 0.0, 0.0], filters=MemoryFilter(owner_id="u1"), k=5)]
    assert "expired" not in ids
    assert "never" in ids and "future" in ids


def test_vector_search_include_expired():
    b = _b()
    b.upsert([_rec("expired", 100)])
    ids = [h.record.id for h in b.vector_search(
        [1.0, 0.0, 0.0], filters=MemoryFilter(owner_id="u1", include_expired=True), k=5)]
    assert ids == ["expired"]


def test_purge_expired_removes_rows():
    b = _b()
    b.upsert([_rec("never", 0), _rec("expired", 100), _rec("future", 5000)])
    n = b.purge_expired(1000)
    assert n == 1
    assert b.get(["expired"]) == []
    assert len(b.get(["never"])) == 1 and len(b.get(["future"])) == 1


def test_keyword_search_excludes_expired():
    b = _b()   # 文件顶部已有：SqliteVecBackend(":memory:", dimension=3, now_fn=lambda: 1000)
    r = MemoryRecord(owner_id="u1", kind="k", mem_type=MemType.SEMANTIC,
                     text="排序算法讲解", embedding=[1.0, 0.0, 0.0], id="e")
    r.expires_at = 100          # 已过期（<= now=1000）
    b.upsert([r])
    assert b.keyword_search("排序算法", filters=MemoryFilter(owner_id="u1"), k=5) == []
    incl = b.keyword_search("排序算法",
                            filters=MemoryFilter(owner_id="u1", include_expired=True), k=5)
    assert len(incl) == 1       # include_expired=True 时可见
