# tests/test_sqlite_backend.py
import pytest

from harness.memory.record import MemType, MemoryFilter, MemoryRecord
from harness.memory.sqlite_backend import SqliteVecBackend


def _rec(owner, kind, text, vec, mem_type=MemType.SEMANTIC, entity_key="", rid=None):
    r = MemoryRecord(owner_id=owner, kind=kind, mem_type=mem_type,
                     text=text, embedding=vec, entity_key=entity_key)
    if rid:
        r.id = rid
    return r


def _backend():
    return SqliteVecBackend(":memory:", dimension=3)


def test_partition_isolation():
    b = _backend()
    b.upsert([_rec("u1", "knowledge", "u1 secret", [1.0, 0.0, 0.0])])
    b.upsert([_rec("u2", "knowledge", "u2 secret", [1.0, 0.0, 0.0])])
    hits = b.vector_search([1.0, 0.0, 0.0], filters=MemoryFilter(owner_id="u2"), k=5)
    assert [h.record.text for h in hits] == ["u2 secret"]   # 绝不召回 u1


def test_kind_filter_pushdown():
    b = _backend()
    b.upsert([_rec("u1", "knowledge", "kb", [1.0, 0.0, 0.0])])
    b.upsert([_rec("u1", "conversation", "conv", [1.0, 0.0, 0.0])])
    hits = b.vector_search([1.0, 0.0, 0.0],
                           filters=MemoryFilter(owner_id="u1", kind="conversation"), k=5)
    assert [h.record.text for h in hits] == ["conv"]


def test_mem_type_filter_pushdown():
    b = _backend()
    b.upsert([_rec("u1", "k", "sem", [1.0, 0.0, 0.0], mem_type=MemType.SEMANTIC)])
    b.upsert([_rec("u1", "k", "epi", [1.0, 0.0, 0.0], mem_type=MemType.EPISODIC)])
    hits = b.vector_search([1.0, 0.0, 0.0],
                           filters=MemoryFilter(owner_id="u1", mem_type="episodic"), k=5)
    assert [h.record.text for h in hits] == ["epi"]


def test_upsert_overwrites_same_id():
    b = _backend()
    b.upsert([_rec("u1", "k", "v1", [1.0, 0.0, 0.0], rid="fixed")])
    b.upsert([_rec("u1", "k", "v2", [1.0, 0.0, 0.0], rid="fixed")])
    got = b.get(["fixed"])
    assert len(got) == 1 and got[0].text == "v2"


def test_delete_removes_from_both_tables():
    b = _backend()
    b.upsert([_rec("u1", "k", "x", [1.0, 0.0, 0.0], rid="d1")])
    b.delete(["d1"])
    assert b.get(["d1"]) == []
    assert b.vector_search([1.0, 0.0, 0.0], filters=MemoryFilter(owner_id="u1"), k=5) == []


def test_list_by_entity():
    b = _backend()
    b.upsert([_rec("u1", "k", "lang=zh", [1.0, 0.0, 0.0], entity_key="pref.lang")])
    b.upsert([_rec("u1", "k", "unrelated", [0.0, 1.0, 0.0], entity_key="")])
    recs = b.list_by_entity("u1", "k", "pref.lang")
    assert [r.text for r in recs] == ["lang=zh"]


def test_superseded_excluded_by_default():
    b = _backend()
    r = _rec("u1", "k", "old", [1.0, 0.0, 0.0], rid="s1")
    r.superseded = 1
    b.upsert([r])
    assert b.vector_search([1.0, 0.0, 0.0], filters=MemoryFilter(owner_id="u1"), k=5) == []
    incl = b.vector_search([1.0, 0.0, 0.0],
                           filters=MemoryFilter(owner_id="u1", include_superseded=True), k=5)
    assert [h.record.text for h in incl] == ["old"]


def test_dimension_mismatch_raises():
    b = _backend()
    with pytest.raises(ValueError):
        b.upsert([_rec("u1", "k", "bad", [1.0, 0.0])])   # 2 维，期望 3


def test_upsert_batch_is_atomic_on_bad_dim():
    b = _backend()
    with pytest.raises(ValueError):
        b.upsert([_rec("u1", "k", "good", [1.0, 0.0, 0.0], rid="g1"),
                  _rec("u1", "k", "bad", [1.0, 0.0])])   # 第二条维度错
    assert b.get(["g1"]) == []          # 整批未生效，前面合法记录也不应残留

