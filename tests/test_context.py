from harness.context.manager import ContextManager
from harness.state import RunState
from harness.types import Message, Role


def test_build_prepends_system_prompt():
    cm = ContextManager(system_prompt="你是助手")
    state = RunState(run_id="r1")
    state.append(Message(role=Role.USER, content="hi"))
    built = cm.build(state)
    assert built[0].role == Role.SYSTEM
    assert built[0].content == "你是助手"
    assert built[1].content == "hi"
    assert len(built) == 2


def test_build_is_pure_does_not_mutate_state():
    cm = ContextManager(system_prompt="s")
    state = RunState(run_id="r1")
    state.append(Message(role=Role.USER, content="hi"))
    cm.build(state)
    assert len(state.messages) == 1  # 未被污染
