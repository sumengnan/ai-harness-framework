# tests/test_windowing_tool_pairing.py
"""锁定保证：layered 上下文的窗口截断绝不拆开「assistant(tool_calls) ↔ tool 结果」配对，
也不会让 kept 以孤儿 tool 开头（否则 OpenAI 会 400）。这是多轮工具任务（如考试回放已抽题目）
在 layered 模式下的正确性前提。"""
from harness.context.windowing import WindowStrategy
from harness.types import Message, Role, ToolCall


def _turn(q: str, tid: str) -> list[Message]:
    # 一轮 = user + assistant(tool_calls) + tool 结果 + assistant 最终答复（回放后的形状）
    return [
        Message(role=Role.USER, content=q),
        Message(role=Role.ASSISTANT,
                tool_calls=[ToolCall(id=tid, name="sample_questions", arguments={"count": 10})]),
        Message(role=Role.TOOL, tool_call_id=tid, content="[题1…题10]"),
        Message(role=Role.ASSISTANT, content="第1题"),
    ]


def _assert_pairing_valid(msgs: list[Message]) -> None:
    assert not msgs or msgs[0].role != Role.TOOL, "kept 不能以孤儿 tool 开头"
    open_ids: set[str] = set()
    for m in msgs:
        if m.role == Role.TOOL:
            assert m.tool_call_id in open_ids, "孤儿 tool 结果（无前置 assistant tool_calls）"
        if m.role == Role.ASSISTANT and m.tool_calls:
            open_ids.update(tc.id for tc in m.tool_calls)


def test_window_never_splits_tool_pairing_across_budgets():
    history = _turn("q1", "t1") + _turn("q2", "t2") + _turn("q3", "t3")
    ws = WindowStrategy("gpt-4o-mini")
    # 任意预算下，kept 都必须配对完整，且从 user 边界起（或为空）
    for budget in (1, 10, 25, 60, 200, 1000, 100000):
        res = ws.select(history, budget)
        _assert_pairing_valid(res.kept)
        assert not res.kept or res.kept[0].role == Role.USER
        # evicted + kept 必须无损覆盖原历史（不丢不重）
        assert len(res.kept) + len(res.evicted) == len(history)


def test_window_strips_leading_orphan_tool():
    # 异常历史：开头就是孤儿 tool（无前置 tool_calls）→ 必须被剥离
    history = [Message(role=Role.TOOL, tool_call_id="x", content="r"),
               Message(role=Role.USER, content="q"),
               Message(role=Role.ASSISTANT, content="a")]
    res = WindowStrategy("gpt-4o-mini").select(history, 100000)
    assert res.kept[0].role != Role.TOOL
