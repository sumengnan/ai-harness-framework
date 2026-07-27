import pytest

from harness.loop.agent_loop import AgentLoop, _accumulate, _finalize
from harness.llm.base import ToolCallDelta, StreamChunk
from harness.context.manager import ContextManager
from harness.tools.base import ToolRegistry
from harness.tools.builtins.calculator import CalculatorTool
from harness.events import (
    RunStarted, TextDelta, ToolCallRequested, ToolStarted, ToolFinished,
    RunFinished, RunError, StepStarted,
)
from harness.reliability.budget import BudgetTracker
from harness.events import ModelUsage


def _build_loop(client, max_steps=10):
    reg = ToolRegistry()
    reg.register(CalculatorTool())
    ctx = ContextManager(system_prompt="s")
    return AgentLoop(
        client=client, registry=reg, context=ctx,
        max_steps=max_steps, run_id_factory=lambda: "run-test",
    )


async def _collect(loop, msg):
    return [ev async for ev in loop.run(msg)]


async def test_plain_chat_terminates_without_tools(make_mock, text_turn):
    loop = _build_loop(make_mock([text_turn("你好呀")]))
    events = await _collect(loop, "hi")
    assert isinstance(events[0], RunStarted)
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "你好呀"
    assert isinstance(events[-1], RunFinished)
    assert events[-1].message.content == "你好呀"
    # 未触发任何工具
    assert not any(isinstance(e, ToolStarted) for e in events)


def _loop_with_detect(client, window, max_steps=10):
    reg = ToolRegistry(); reg.register(CalculatorTool())
    return AgentLoop(client=client, registry=reg,
                     context=ContextManager(system_prompt="s"),
                     max_steps=max_steps, run_id_factory=lambda: "run-test",
                     loop_detect_window=window)


async def test_loop_nudge_gives_model_another_chance(make_mock, tool_turn, text_turn):
    # 连续 window 步相同 → 注入纠偏（不中止）；模型下一步改口给正文 → 正常收尾
    client = make_mock([
        tool_turn("calculator", '{"expression": "1+1"}', call_id="c1"),
        tool_turn("calculator", '{"expression": "1+1"}', call_id="c1"),
        text_turn("好了"),
    ])
    loop = _loop_with_detect(client, window=2)
    events = await _collect(loop, "算")
    assert isinstance(events[-1], RunFinished)          # 纠偏给了机会，未过早中止
    assert events[-1].message.content == "好了"
    # step1 在窗口填满前先执行了工具（1 次）；step2 命中纠偏、不真执行；step3 收尾
    assert sum(isinstance(e, ToolFinished) for e in events) == 1


async def test_loop_aborts_after_nudge_still_repeating(make_mock, tool_turn):
    # 纠偏后模型仍一味重复相同调用 → 最终中止
    turns = [tool_turn("calculator", '{"expression": "1+1"}', call_id="c1") for _ in range(6)]
    loop = _loop_with_detect(make_mock(turns), window=2)
    events = await _collect(loop, "算")
    assert isinstance(events[-1], RunError)
    assert "纠偏后仍" in events[-1].error


async def test_loop_detection_disabled_when_window_lt_2(make_mock, tool_turn, text_turn):
    # window<2 关闭：相同工具调用不触发中止，正常收尾
    client = make_mock([
        tool_turn("calculator", '{"expression": "1+1"}', call_id="c1"),
        tool_turn("calculator", '{"expression": "1+1"}', call_id="c1"),
        text_turn("好了"),
    ])
    loop = _loop_with_detect(client, window=0)
    events = await _collect(loop, "算")
    assert isinstance(events[-1], RunFinished)


async def test_varied_args_not_flagged_as_loop(make_mock, tool_turn, text_turn):
    # 同工具但每步参数不同（如翻页）→ 签名不同、不判循环
    client = make_mock([
        tool_turn("calculator", '{"expression": "1+1"}', call_id="c1"),
        tool_turn("calculator", '{"expression": "2+2"}', call_id="c1"),
        tool_turn("calculator", '{"expression": "3+3"}', call_id="c1"),
        text_turn("完"),
    ])
    loop = _loop_with_detect(client, window=3)
    events = await _collect(loop, "算")
    assert isinstance(events[-1], RunFinished)


async def test_tool_call_executes_and_feeds_back(make_mock, text_turn, tool_turn):
    client = make_mock([
        tool_turn("calculator", '{"expression": "(12+8)*3"}', call_id="c1"),
        text_turn("答案是 60"),
    ])
    loop = _build_loop(client)
    events = await _collect(loop, "算 (12+8)*3")
    assert any(isinstance(e, ToolCallRequested) for e in events)
    finished = [e for e in events if isinstance(e, ToolFinished)]
    assert len(finished) == 1
    assert finished[0].result.content == "60"
    assert finished[0].result.is_error is False
    assert isinstance(events[-1], RunFinished)
    assert events[-1].message.content == "答案是 60"


async def test_max_steps_guard_emits_run_error(make_mock, tool_turn):
    # 每轮都请求工具，永不给最终答案 → 应在 max_steps 后 RunError
    turns = [tool_turn("calculator", '{"expression": "1+1"}', call_id=f"c{i}")
             for i in range(5)]
    loop = _build_loop(make_mock(turns), max_steps=2)
    events = await _collect(loop, "loop forever")
    assert isinstance(events[-1], RunError)
    assert "max_steps" in events[-1].error
    # 恰好 2 个 StepStarted
    assert sum(isinstance(e, StepStarted) for e in events) == 2


async def test_llm_stream_exception_becomes_run_error(text_turn):
    class BoomClient:
        async def stream(self, messages, tools):
            raise ConnectionError("network down")
            yield  # pragma: no cover  (使其成为 async generator)

    loop = _build_loop(BoomClient())
    events = await _collect(loop, "hi")
    assert isinstance(events[-1], RunError)
    assert "network down" in events[-1].error


async def test_bad_tool_args_feed_back_is_error(make_mock, text_turn, tool_turn):
    # 参数缺 expression → executor 返回 is_error，loop 照常回填并继续
    client = make_mock([
        tool_turn("calculator", '{}', call_id="c1"),
        text_turn("我需要一个表达式"),
    ])
    loop = _build_loop(client)
    events = await _collect(loop, "算点啥")
    finished = [e for e in events if isinstance(e, ToolFinished)]
    assert finished[0].result.is_error is True
    assert isinstance(events[-1], RunFinished)


def _build_loop_with_budget(client, budget, max_steps=10):
    reg = ToolRegistry()
    reg.register(CalculatorTool())
    ctx = ContextManager(system_prompt="s")
    return AgentLoop(client=client, registry=reg, context=ctx, max_steps=max_steps,
                     run_id_factory=lambda: "run-test", budget=budget)


def test_finalize_returns_calls_and_parse_error():
    # 交错双工具 + 一个非法 JSON，验证新的 _Finalized 返回
    acc = {}
    _accumulate(acc, ToolCallDelta(index=0, id="a", name="calculator", arguments='{"expression":'))
    _accumulate(acc, ToolCallDelta(index=1, id="b", name="echo", arguments='{"text":"hi"}'))
    _accumulate(acc, ToolCallDelta(index=0, arguments='"1+1"}'))
    out = _finalize(acc)
    assert [f.call.name for f in out] == ["calculator", "echo"]
    assert out[0].call.arguments == {"expression": "1+1"}
    assert out[0].parse_error is None


def test_finalize_flags_invalid_json():
    acc = {}
    _accumulate(acc, ToolCallDelta(index=0, id="a", name="echo", arguments="not-json"))
    out = _finalize(acc)
    assert out[0].parse_error is not None
    assert out[0].call.arguments == {}


async def test_invalid_json_tool_args_self_correct(make_mock, text_turn):
    # 第一轮吐非法 JSON 工具参数 → loop 应回填 is_error 错误消息，不崩溃；第二轮作答
    from harness.llm.base import StreamChunk
    bad_tool_turn = [
        StreamChunk(type="tool_call", tool_call_delta=ToolCallDelta(
            index=0, id="c1", name="calculator", arguments="not-json")),
        StreamChunk(type="done"),
    ]
    loop = _build_loop(make_mock([bad_tool_turn, text_turn("抱歉，我重发")]))
    from harness.events import ToolFinished, RunFinished
    events = [e async for e in loop.run("算点啥")]
    finished = [e for e in events if isinstance(e, ToolFinished)]
    assert finished[0].result.is_error is True
    assert "JSON" in finished[0].result.content
    assert isinstance(events[-1], RunFinished)


async def test_token_budget_breach_emits_run_error(make_mock):
    # 预算 50，每轮请求工具且 usage=40：step1 后累计 40，step2 后 80，step3 步边界拦截
    from harness.events import RunError
    from harness.usage import Usage
    from harness.llm.base import StreamChunk

    def tool_usage_turn(i):
        return [
            StreamChunk(type="tool_call", tool_call_delta=ToolCallDelta(
                index=0, id=f"c{i}", name="calculator", arguments='{"expression":"1+1"}')),
            StreamChunk(type="done", usage=Usage(20, 20, 40)),
        ]

    loop = _build_loop_with_budget(make_mock([tool_usage_turn(i) for i in range(5)]),
                                   BudgetTracker(max_tokens=50))
    events = [e async for e in loop.run("go")]
    assert isinstance(events[-1], RunError)
    assert "token" in events[-1].error


async def test_model_usage_event_emitted(make_mock, text_turn_usage):
    from harness.events import ModelUsage
    loop = _build_loop(make_mock([text_turn_usage("你好", prompt=10, completion=5)]))
    events = [e async for e in loop.run("hi")]
    mu = [e for e in events if isinstance(e, ModelUsage)]
    assert len(mu) == 1
    assert mu[0].usage.total_tokens == 15


def test_finalize_flags_non_dict_json():
    # 合法 JSON 但不是对象（数组）→ parse_error 非空、arguments 降级为空 dict
    acc = {}
    _accumulate(acc, ToolCallDelta(index=0, id="a", name="echo", arguments="[1,2]"))
    out = _finalize(acc)
    assert out[0].parse_error is not None
    assert out[0].call.arguments == {}


async def test_retry_exhausted_becomes_run_error(flaky_client, text_turn):
    # 重试耗尽 → RetryingModelClient 抛瞬时错误 → loop 的 except → RunError（不外泄异常）
    from harness.reliability.retry import RetryingModelClient

    class Transient(Exception):
        pass

    async def fake_sleep(d):
        pass

    inner = flaky_client(Transient("timeout"), text_turn("never"), fail_times=99)
    client = RetryingModelClient(inner, max_retries=2, base_delay=0.01,
                                 sleep=fake_sleep, transient=(Transient,))
    loop = _build_loop(client)
    events = [e async for e in loop.run("hi")]
    assert isinstance(events[-1], RunError)
    assert "模型调用失败" in events[-1].error


class _FakeClock:
    """按调用次数返回序列值；越界后返回最后一个值，避免 StopIteration。"""

    def __init__(self, seq):
        self._seq = seq
        self._i = 0

    def __call__(self):
        v = self._seq[min(self._i, len(self._seq) - 1)]
        self._i += 1
        return v


async def test_wall_budget_breach_emits_run_error(make_mock):
    from harness.usage import Usage

    def tool_usage_turn(i):
        return [
            StreamChunk(type="tool_call", tool_call_delta=ToolCallDelta(
                index=0, id=f"c{i}", name="calculator", arguments='{"expression":"1+1"}')),
            StreamChunk(type="done", usage=Usage(1, 1, 2)),
        ]

    # 时钟调用序列：start()=0.0, step1 check=1.0（未超）, step2 check=5.0（>3.0 超限）
    clock = _FakeClock([0.0, 1.0, 5.0])
    budget = BudgetTracker(max_wall_seconds=3.0, clock=clock)
    loop = _build_loop_with_budget(make_mock([tool_usage_turn(i) for i in range(5)]), budget)
    events = [e async for e in loop.run("go")]
    assert isinstance(events[-1], RunError)
    assert "时间" in events[-1].error
