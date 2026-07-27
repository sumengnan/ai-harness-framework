# tests/test_backend_superseded.py
from harness.memory.record import MemType, MemoryFilter, MemoryRecord
from harness.memory.sqlite_backend import SqliteVecBackend


def _rec(rid, vec=(1.0, 0.0, 0.0)):
    return MemoryRecord(owner_id="u1", kind="k", mem_type=MemType.SEMANTIC,
                        text=f"fact {rid}", embedding=list(vec), id=rid)


def test_set_superseded_excludes_from_search():
    b = SqliteVecBackend(":memory:", dimension=3)
    b.upsert([_rec("a"), _rec("b", (0.0, 1.0, 0.0))])
    b.set_superseded(["a"])
    hits = b.vector_search([1.0, 0.0, 0.0], filters=MemoryFilter(owner_id="u1"), k=5)
    assert "a" not in [h.record.id for h in hits]
    incl = b.vector_search([1.0, 0.0, 0.0],
                           filters=MemoryFilter(owner_id="u1", include_superseded=True), k=5)
    assert "a" in [h.record.id for h in incl]


def test_set_superseded_get_still_returns():
    b = SqliteVecBackend(":memory:", dimension=3)
    b.upsert([_rec("a")])
    b.set_superseded(["a"])
    got = b.get(["a"])
    assert len(got) == 1 and got[0].superseded == 1


def test_set_superseded_missing_id_noop():
    b = SqliteVecBackend(":memory:", dimension=3)
    b.set_superseded(["nope"])
