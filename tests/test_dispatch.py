import pytest

from harness.orchestration.spec import AgentSpec, AgentRoster
from harness.orchestration.dispatch import DispatchTool
from harness.tools.base import ToolRegistry, ToolExecutor
from harness.tools.builtins.calculator import CalculatorTool
from harness.types import ToolCall


def _dispatch(client, depth=0, max_depth=2, tools=("calculator",)):
    roster = AgentRoster([AgentSpec("researcher", "研究员", "你是研究员", list(tools))])
    pool = {"calculator": CalculatorTool()}
    return DispatchTool(roster, pool, client, depth=depth, max_depth=max_depth, sub_max_steps=5)


async def test_dispatch_runs_subagent_and_returns_answer(make_mock, text_turn):
    tool = _dispatch(make_mock([text_turn("子结果")]))
    out = await tool.run(tool.Params(agent="researcher", task="做点研究"))
    assert out == "子结果"


async def test_dispatch_emits_subagent_progress(make_mock, tool_turn, text_turn):
    from harness import progress
    from harness.events import Progress
    tool = _dispatch(make_mock([tool_turn("calculator", '{"expression":"1+1"}'),
                                text_turn("子结果")]))
    got = []
    token = progress.set_emitter(got.append)
    try:
        out = await tool.run(tool.Params(agent="researcher", task="做点研究"))
    finally:
        progress.reset_emitter(token)
    assert out == "子结果"
    prog = [(e.scope, e.text) for e in got if isinstance(e, Progress)]
    assert prog and all(s == "subagent:researcher" for s, _ in prog)
    texts = [t for _, t in prog]
    assert any("开始任务" in t for t in texts)
    assert any("调用工具 calculator" in t for t in texts)
    assert any("任务完成" in t for t in texts)


async def test_dispatch_tool_progress_carries_detail(make_mock, tool_turn, text_turn):
    """dispatch 子代理工具行带 detail：开始行 tool+args，完成行 result+is_error 且保留 args。"""
    from harness import progress
    from harness.events import Progress
    tool = _dispatch(make_mock([tool_turn("calculator", '{"expression":"1+1"}'),
                                text_turn("子结果")]))
    got = []
    token = progress.set_emitter(got.append)
    try:
        await tool.run(tool.Params(agent="researcher", task="做点研究"))
    finally:
        progress.reset_emitter(token)
    tool_rows = [e for e in got if isinstance(e, Progress) and e.detail is not None]
    assert any(r.detail["tool"] == "calculator" and r.detail["args"] == {"expression": "1+1"}
               for r in tool_rows)
    finished = [r for r in tool_rows if "result" in r.detail]
    assert finished and finished[-1].detail["args"] == {"expression": "1+1"}
    assert "is_error" in finished[-1].detail


async def test_unknown_agent_is_error(make_mock, text_turn):
    tool = _dispatch(make_mock([text_turn("x")]))
    reg = ToolRegistry(); reg.register(tool)
    ex = ToolExecutor(reg)
    r = await ex.execute(ToolCall(id="c1", name="dispatch",
                                  arguments={"agent": "nobody", "task": "t"}))
    assert r.is_error is True
    assert "未知角色" in r.content


def test_sub_registry_only_spec_tools(make_mock, text_turn):
    tool = _dispatch(make_mock([text_turn("x")]), tools=("calculator",))
    spec = tool._roster.get("researcher")
    reg = tool._build_sub_registry(spec)
    assert reg.get("calculator") is not None
    assert reg.get("browse") is None            # 未列的工具不在


def test_depth_limit_controls_dispatch_injection(make_mock, text_turn):
    spec = AgentSpec("researcher", "r", "p", ["calculator"])
    d0 = _dispatch(make_mock([text_turn("x")]), depth=0, max_depth=2)
    assert d0._build_sub_registry(spec).get("dispatch") is not None   # depth+1=1<2 → 含
    d1 = _dispatch(make_mock([text_turn("x")]), depth=1, max_depth=2)
    assert d1._build_sub_registry(spec).get("dispatch") is None       # depth+1=2 不<2 → 不含


async def test_dispatch_description_lists_roles(make_mock, text_turn):
    tool = _dispatch(make_mock([text_turn("x")]))
    assert "researcher" in tool.description


async def test_subagent_no_result_is_error(make_mock, tool_turn):
    # 子 agent 只吐工具调用、sub_max_steps=1 → 永不 RunFinished → 达上限 RunError
    # → DispatchTool.run final=None → 抛 RuntimeError → ToolExecutor 兜成 is_error
    roster = AgentRoster([AgentSpec("researcher", "研究员", "你是研究员", ["calculator"])])
    pool = {"calculator": CalculatorTool()}
    client = make_mock([tool_turn("calculator", '{"expression": "1+1"}')])
    tool = DispatchTool(roster, pool, client, depth=0, max_depth=2, sub_max_steps=1)
    reg = ToolRegistry(); reg.register(tool)
    ex = ToolExecutor(reg)
    r = await ex.execute(ToolCall(id="c1", name="dispatch",
                                  arguments={"agent": "researcher", "task": "t"}))
    assert r.is_error is True
    assert "未产出结果" in r.content
