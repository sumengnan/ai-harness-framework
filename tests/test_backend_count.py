# tests/test_backend_count.py
from harness.memory.record import MemoryRecord, MemType
from harness.memory.sqlite_backend import SqliteVecBackend


def _b():
    return SqliteVecBackend(":memory:", dimension=3)


def _rec(rid, mem_type, owner="u1", kind="k"):
    return MemoryRecord(id=rid, owner_id=owner, kind=kind, mem_type=mem_type,
                        text=f"t-{rid}", embedding=[1.0, 0.0, 0.0])


def test_count_by_owner_counts_all_when_no_mem_type():
    b = _b()
    b.upsert([_rec("a", MemType.EPISODIC), _rec("b", MemType.SEMANTIC),
              _rec("c", MemType.PROCEDURAL)])
    assert b.count_by_owner("u1", "k") == 3          # 既有调用方（知识库数总数）语义不变


def test_count_by_owner_filters_by_mem_type():
    """记忆整合据此判断该不该触发：只吃 episodic，用总数会让零 episodic 的会话每轮空转。"""
    b = _b()
    b.upsert([_rec("a", MemType.EPISODIC), _rec("b", MemType.EPISODIC),
              _rec("c", MemType.SEMANTIC)])
    assert b.count_by_owner("u1", "k", mem_type=MemType.EPISODIC) == 2
    assert b.count_by_owner("u1", "k", mem_type=MemType.SEMANTIC) == 1
    assert b.count_by_owner("u1", "k", mem_type="episodic") == 2    # 字符串亦可


def test_count_by_owner_excludes_superseded():
    """整合把 episodic 标 superseded 后计数须回落，否则会每轮重触发。"""
    b = _b()
    b.upsert([_rec("a", MemType.EPISODIC), _rec("b", MemType.EPISODIC)])
    b.set_superseded(["a"])
    assert b.count_by_owner("u1", "k", mem_type=MemType.EPISODIC) == 1


def test_count_by_owner_isolates_owner_and_kind():
    b = _b()
    b.upsert([_rec("a", MemType.EPISODIC),
              _rec("b", MemType.EPISODIC, owner="u2"),
              _rec("c", MemType.EPISODIC, kind="other")])
    assert b.count_by_owner("u1", "k", mem_type=MemType.EPISODIC) == 1
