from harness.state import RunState
from harness.types import Message, Role


def test_runstate_append_accumulates():
    state = RunState(run_id="r1")
    assert state.messages == []
    assert state.step == 0
    state.append(Message(role=Role.USER, content="hi"))
    state.append(Message(role=Role.ASSISTANT, content="yo"))
    assert [m.content for m in state.messages] == ["hi", "yo"]
