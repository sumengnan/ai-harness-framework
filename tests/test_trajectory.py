from harness.persistence.trajectory import TrajectoryStore, TrajectorySink
from harness.events import RunStarted, TextDelta, RunFinished
from harness.types import Message, Role


async def _gen(evs):
    for e in evs:
        yield e


def test_store_append_load_ordered():
    s = TrajectoryStore(":memory:")
    s.append("r1", 0, {"type": "A", "data": {}})
    s.append("r1", 1, {"type": "B", "data": {}})
    assert [e["type"] for e in s.load("r1")] == ["A", "B"]


def test_next_seq():
    s = TrajectoryStore(":memory:")
    assert s.next_seq("r1") == 0                # 空 → 0
    s.append("r1", 0, {"type": "A", "data": {}})
    s.append("r1", 1, {"type": "B", "data": {}})
    assert s.next_seq("r1") == 2                # MAX(seq)+1
    assert s.next_seq("other") == 0            # 按 run_id 隔离


def test_list_run_ids():
    s = TrajectoryStore(":memory:")
    s.append("r1", 0, {"type": "A", "data": {}})
    s.append("r2", 0, {"type": "A", "data": {}})
    s.append("r1", 1, {"type": "B", "data": {}})
    assert sorted(s.list_run_ids()) == ["r1", "r2"]


def test_delete_only_target_run():
    s = TrajectoryStore(":memory:")
    s.append("r1", 0, {"type": "A", "data": {}})
    s.append("r1", 1, {"type": "B", "data": {}})
    s.append("r2", 0, {"type": "A", "data": {}})
    s.delete("r1")
    assert s.load("r1") == []                   # 目标 run 清空
    assert [e["type"] for e in s.load("r2")] == ["A"]   # 其他 run 不受影响
    assert s.list_run_ids() == ["r2"]


async def test_sink_records_and_passes_through():
    store = TrajectoryStore(":memory:")
    sink = TrajectorySink(store)
    evs = [RunStarted(run_id="r1"), TextDelta(text="hi"),
           RunFinished(message=Message(role=Role.ASSISTANT, content="done"))]
    out = [e async for e in sink.wrap(_gen(evs))]
    assert len(out) == 3                               # 透传
    assert [e["type"] for e in store.load("r1")] == ["RunStarted", "TextDelta", "RunFinished"]
