from harness.context.windowing import WindowStrategy
from harness.types import Message, Role, ToolCall
from harness.usage import count_message_tokens

_M = "gpt-4o-mini"


def _turn(u: str, a: str) -> list[Message]:
    return [Message(role=Role.USER, content=u), Message(role=Role.ASSISTANT, content=a)]


def _tool_turn() -> list[Message]:
    """一个含工具调用的完整轮：user → assistant(tool_calls) → tool → assistant。"""
    return [
        Message(role=Role.USER, content="算一下 1..100 的和"),
        Message(role=Role.ASSISTANT, content=None,
                tool_calls=[ToolCall(id="c1", name="run_python",
                                     arguments={"code": "print(sum(range(101)))"})]),
        Message(role=Role.TOOL, content="5050", tool_call_id="c1"),
        Message(role=Role.ASSISTANT, content="结果是 5050"),
    ]


def test_short_history_all_kept():
    hist = _turn("你好", "你好呀") + _turn("再见", "回见")
    r = WindowStrategy(_M).select(hist, budget_tokens=100_000)
    assert r.kept == hist
    assert r.evicted == []


def test_over_budget_drops_old_turns_at_user_boundary():
    hist = _turn("轮1问", "轮1答") + _turn("轮2问", "轮2答") + _turn("轮3问", "轮3答")
    # 预算只够最后两轮
    budget = count_message_tokens(hist[2:], _M) + 5
    r = WindowStrategy(_M).select(hist, budget_tokens=budget)
    assert r.kept[0].role == Role.USER          # 干净的 user 边界
    assert r.kept[0].content == "轮2问"
    assert r.evicted == hist[:2]
    assert count_message_tokens(r.kept, _M) <= budget


def test_tool_call_turn_not_split():
    hist = _turn("闲聊", "嗯") + _tool_turn()
    # 预算只够最后那个工具轮
    budget = count_message_tokens(_tool_turn(), _M) + 5
    r = WindowStrategy(_M).select(hist, budget_tokens=budget)
    # 保留的窗口首条必须是 user，且 tool 结果不能成为孤儿
    assert r.kept[0].role == Role.USER
    tool_ids = {m.tool_call_id for m in r.kept if m.role == Role.TOOL}
    call_ids = {tc.id for m in r.kept for tc in m.tool_calls}
    assert tool_ids <= call_ids  # 每个 tool 结果都有对应的 tool_call


def test_single_turn_exceeds_budget_still_kept():
    hist = _turn("很长很长的一轮问题" * 50, "很长很长的一轮回答" * 50)
    r = WindowStrategy(_M).select(hist, budget_tokens=1)
    assert r.kept == hist          # 至少保留最后一个完整轮
    assert r.evicted == []


def test_no_user_messages_keeps_all():
    hist = [Message(role=Role.ASSISTANT, content="系统播报")]
    r = WindowStrategy(_M).select(hist, budget_tokens=1)
    assert r.kept == hist


def test_select_does_not_mutate_input():
    hist = _turn("a", "b") + _turn("c", "d")
    original = list(hist)
    WindowStrategy(_M).select(hist, budget_tokens=1)
    assert hist == original
