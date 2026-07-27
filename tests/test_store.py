import pytest

from harness.memory.store import MemoryStore, MemoryHit


def _store():
    return MemoryStore(":memory:", dimension=4)


def test_add_and_search_returns_nearest():
    s = _store()
    s.add([
        ("knowledge", "a", {"source": "s1"}, [1.0, 0.0, 0.0, 0.0]),
        ("knowledge", "b", {}, [0.0, 1.0, 0.0, 0.0]),
        ("knowledge", "c", {}, [0.0, 0.0, 1.0, 0.0]),
    ])
    hits = s.search("knowledge", [1.0, 0.0, 0.0, 0.0], k=2)
    assert hits[0].text == "a"                 # 最近
    assert hits[0].metadata["source"] == "s1"
    assert len(hits) == 2
    assert isinstance(hits[0], MemoryHit)


def test_collection_isolation():
    s = _store()
    s.add([
        ("knowledge", "k1", {}, [1.0, 0.0, 0.0, 0.0]),
        ("notes", "n1", {}, [1.0, 0.0, 0.0, 0.0]),
    ])
    hits = s.search("notes", [1.0, 0.0, 0.0, 0.0], k=5)
    assert [h.text for h in hits] == ["n1"]    # 不串 collection


def test_delete():
    s = _store()
    ids = s.add([("knowledge", "x", {}, [1.0, 0.0, 0.0, 0.0])])
    s.delete(ids)
    assert s.search("knowledge", [1.0, 0.0, 0.0, 0.0], k=5) == []


def test_adaptive_overfetch_recalls_target_collection():
    # 其他 collection 挤占前 k*4：固定 over-fetch 会零召回，自适应应仍返回 k 条 knowledge
    s = _store()
    items = [("knowledge", f"k{i}", {}, [1.0, 0.0, 0.0, 0.0]) for i in range(3)]
    # other 向量更接近 query（完全相同方向），会排在最前挤占名额
    items += [("other", f"o{i}", {}, [1.0, 0.0, 0.0, 0.0]) for i in range(20)]
    s.add(items)
    hits = s.search("knowledge", [1.0, 0.0, 0.0, 0.0], k=3)
    assert len(hits) == 3
    assert all(h.collection == "knowledge" for h in hits)


def test_add_dimension_mismatch_raises_value_error():
    s = _store()
    with pytest.raises(ValueError):
        s.add([("knowledge", "bad", {}, [1.0, 0.0, 0.0])])  # 只有 3 维，期望 4 维
