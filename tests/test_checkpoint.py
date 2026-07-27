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
