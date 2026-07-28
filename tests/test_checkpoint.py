from harness.persistence.checkpoint import CheckpointStore
from harness.state import RunState
from harness.types import Message, Role


def test_save_load_roundtrip():
    cs = CheckpointStore(":memory:")
    st = RunState(run_id="r1"); st.step = 3
    st.append(Message(role=Role.USER, content="hi"))
    cs.save(st)
    loaded = cs.load("r1")
    assert loaded.run_id == "r1" and loaded.step == 3 and loaded.messages[0].content == "hi"


def test_load_missing_returns_none():
    assert CheckpointStore(":memory:").load("nope") is None


def test_delete():
    cs = CheckpointStore(":memory:")
    cs.save(RunState(run_id="r1"))
    cs.delete("r1")
    assert cs.load("r1") is None


def test_usable_from_another_thread(tmp_path):
    """agent 常跑在线程池或另起的事件循环里，共用 store 实例不能炸。"""
    import threading

    cs = CheckpointStore(str(tmp_path / "c.db"))
    cs.save(RunState(run_id="r1"))

    result, errors = [], []

    def worker():
        try:
            st = cs.load("r1")
            cs.save(RunState(run_id="r2"))
            result.append(st.run_id)
        except Exception as e:                 # noqa: BLE001 — 要的就是"任何异常都算失败"
            errors.append(repr(e))

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert errors == []
    assert result == ["r1"]
    assert cs.load("r2") is not None           # 别的线程写的也读得到


def test_concurrent_writes_serialized(tmp_path):
    """多线程并发写同一个 store：不丢不炸。"""
    import threading

    cs = CheckpointStore(str(tmp_path / "c.db"))
    errors = []

    def writer(i):
        try:
            for _ in range(20):
                st = RunState(run_id=f"r{i}")
                st.step = i
                cs.save(st)
        except Exception as e:                 # noqa: BLE001
            errors.append(repr(e))

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert [cs.load(f"r{i}").step for i in range(8)] == list(range(8))


def test_budget_fields_roundtrip():
    cs = CheckpointStore(":memory:")
    st = RunState(run_id="r1")
    st.tokens_used = 1234
    st.wall_seconds_used = 45.5
    cs.save(st)
    loaded = cs.load("r1")
    assert loaded.tokens_used == 1234
    assert loaded.wall_seconds_used == 45.5
