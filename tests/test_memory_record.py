# tests/test_memory_record.py
from harness.memory.record import MemType, MemoryRecord, MemoryFilter, MemoryHit


def test_memtype_values():
    assert MemType.EPISODIC.value == "episodic"
    assert MemType.SEMANTIC.value == "semantic"
    assert MemType.PROCEDURAL.value == "procedural"


def test_record_defaults_and_sentinels():
    r = MemoryRecord(owner_id="u1", kind="knowledge",
                     mem_type=MemType.SEMANTIC, text="hello")
    assert r.id                      # 自动生成非空
    assert r.entity_key == ""        # 哨兵
    assert r.superseded == 0
    assert r.expires_at == 0         # 哨兵：永不过期
    assert r.version == 1
    assert r.importance == 0.5
    assert r.access_count == 0
    assert r.embedding == []
    assert r.created_at and r.last_accessed_at
    assert r.metadata == {}


def test_record_ids_unique():
    a = MemoryRecord(owner_id="u", kind="k", mem_type=MemType.SEMANTIC, text="x")
    b = MemoryRecord(owner_id="u", kind="k", mem_type=MemType.SEMANTIC, text="y")
    assert a.id != b.id


def test_filter_defaults():
    f = MemoryFilter(owner_id="u1")
    assert f.kind is None and f.mem_type is None
    assert f.include_superseded is False and f.include_expired is False


def test_hit_holds_record_and_distance():
    r = MemoryRecord(owner_id="u", kind="k", mem_type=MemType.SEMANTIC, text="t")
    h = MemoryHit(record=r, distance=0.25)
    assert h.record is r and h.distance == 0.25
