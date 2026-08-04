import contextlib

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


from harness.types import Message, Role


async def test_run_accepts_initial_messages(make_mock, text_turn):
    """传入完整初始消息：全部进入状态，且顺序保持。"""
    seen: list[list] = []

    class _Recording:
        def __init__(self, inner):
            self._inner = inner

        async def stream(self, messages, tools):
            seen.append(list(messages))
            async for c in self._inner.stream(messages, tools):
                yield c

    loop = _build_loop(_Recording(make_mock([text_turn("ok")])))
    initial = [
        Message(role=Role.USER, content="诊断如下"),
        Message(role=Role.ASSISTANT, content="收到"),
        Message(role=Role.USER, content="请修复"),
    ]
    events = [ev async for ev in loop.run(messages=initial)]
    assert isinstance(events[-1], RunFinished)
    # ContextManager 在最前插入 system，其后应为传入的三条
    sent = seen[0]
    assert [m.role for m in sent[1:]] == [Role.USER, Role.ASSISTANT, Role.USER]
    assert [m.content for m in sent[1:]] == ["诊断如下", "收到", "请修复"]


async def test_run_still_accepts_plain_string(make_mock, text_turn):
    """旧用法不变。"""
    loop = _build_loop(make_mock([text_turn("ok")]))
    events = [ev async for ev in loop.run("hi")]
    assert isinstance(events[-1], RunFinished)


async def test_run_requires_exactly_one_input(make_mock, text_turn):
    """两者都不给或都给，都是调用方错误。"""
    loop = _build_loop(make_mock([text_turn("ok")]))
    with pytest.raises(ValueError):
        [ev async for ev in loop.run()]
    with pytest.raises(ValueError):
        [ev async for ev in loop.run("hi", messages=[Message(role=Role.USER, content="x")])]


class _SpySpan:
    """记录不了什么，只需接住 _run_from 里对 span 的全部调用，不崩即可。"""

    def set_attribute(self, key, value):
        pass

    def set_status(self, status):
        pass

    def add_event(self, name, attributes=None):
        pass


class _SpyTracer:
    """间谍 tracer：记录每个 span 的进入/退出顺序，用于验证生成器关闭时机。"""

    def __init__(self):
        self.events: list[tuple[str, str]] = []

    @contextlib.contextmanager
    def start_as_current_span(self, name):
        self.events.append(("enter", name))
        try:
            yield _SpySpan()
        finally:
            self.events.append(("exit", name))


async def test_run_aclose_closes_inner_generator(make_mock, tool_turn):
    """壳生成器 run() 被 aclose() 时必须一并终结内层 _run_from。

    否则 run span 与纠偏升温的还原（ExitStack）要等到事件循环回收异步生成器
    才会发生，时机不确定——这正是下游成本闸 await stream.aclose() 撞到的问题。
    """
    spy = _SpyTracer()
    reg = ToolRegistry()
    reg.register(CalculatorTool())
    loop = AgentLoop(
        client=make_mock([tool_turn("calculator", '{"expression": "1+1"}', call_id="c1")]),
        registry=reg, context=ContextManager(system_prompt="s"),
        max_steps=10, run_id_factory=lambda: "run-test", tracer=spy,
    )
    gen = loop.run("hi")
    await gen.__anext__()  # RunStarted：此时还没进入 run span 的 with
    await gen.__anext__()  # StepStarted：run span 已进入，此刻挂起在 with 内部
    assert ("enter", "run") in spy.events
    assert ("exit", "run") not in spy.events  # 还没关，run span 应仍处于打开状态
    await gen.aclose()
    assert ("exit", "run") in spy.events, "关掉外层壳没有终结内层 _run_from"


# ---------------------------------------------------------------- 结果停滞
#
# 签名检测漏掉的那一类：**每次调用都不一样，每次结果都一样**。
#
# 真实来历（2026-08-04，aifix 在 ai-learning-helper 上的一次真跑）：模型九次
# 改同一个函数、九次跑测试，九次拿回**逐字相同**的失败输出。九次的 edit_file
# 参数各不相同，于是签名检测一次都没响，它一路烧到 token 预算耗尽（343K）。
#
# 换着花样撞同一堵墙，和原地不动一样卡住，而且更贵 —— 它看起来像在推进。


def _same_result_turns(tool_turn, n):
    """n 步：表达式每次都不同，算出来的结果都是 2。签名各异、结果相同。"""
    exprs = ["1+1", "0+2", "4-2", "6-4", "1*2", "8-6", "2/1", "3-1"]
    return [tool_turn("calculator", '{"expression": "%s"}' % exprs[i % len(exprs)],
                      call_id=f"c{i}") for i in range(n)]


class _RecordingMock:
    """记下每一轮送进模型的消息 —— 「提醒有没有真的送到模型眼前」只能这样验。

    断言「没有中止」是不够的：什么都不实现时它同样不中止，那条测试会假绿。
    """

    def __init__(self, turns):
        self._turns, self._i = list(turns), 0
        self.seen: list[list] = []

    async def stream(self, messages, tools):
        self.seen.append(list(messages))
        turn = self._turns[self._i]
        self._i += 1
        for chunk in turn:
            yield chunk


async def test_identical_results_from_varied_calls_get_nudged(
        tool_turn, text_turn):
    """连续 N 次拿回逐字相同的结果 → 出声提醒，但**不中止**。

    先提醒不中止，与签名检测同一个形状：模型可能只是需要被告知
    「你做的事没有改变任何东西」，那句话它自己看不出来 —— 而确定性代码
    一次字符串比较就能算出来。
    """
    client = _RecordingMock(_same_result_turns(tool_turn, 3)
                            + [text_turn("换个思路")])
    loop = _loop_with_detect(client, window=3)
    events = await _collect(loop, "算")

    assert isinstance(events[-1], RunFinished)          # 给了机会，没有过早中止
    assert events[-1].message.content == "换个思路"
    # 三次调用都**真的执行了** —— 与签名检测不同，结果要执行完才知道
    assert sum(isinstance(e, ToolFinished) for e in events) == 3

    # 提醒真的进了下一轮的上下文，而且说清了「相同的是结果，不是调用」
    last = "\n".join(str(m.content or "") for m in client.seen[-1])
    assert "系统提示" in last, last
    assert "结果" in last, last
    assert "calculator" in last, last


async def test_identical_results_abort_after_the_nudge(make_mock, tool_turn):
    """提醒之后还是同一个结果 → 中止。不然它会一路烧到 max_steps。"""
    loop = _loop_with_detect(make_mock(_same_result_turns(tool_turn, 10)),
                             window=3)
    events = await _collect(loop, "算")

    assert isinstance(events[-1], RunError)
    assert "结果" in events[-1].error, events[-1].error


async def test_changing_results_are_never_flagged(make_mock, tool_turn, text_turn):
    """结果每次都不同 —— 那是在推进，一个字都不该说。"""
    client = make_mock([
        tool_turn("calculator", '{"expression": "1+1"}', call_id="c1"),
        tool_turn("calculator", '{"expression": "2+2"}', call_id="c2"),
        tool_turn("calculator", '{"expression": "3+3"}', call_id="c3"),
        tool_turn("calculator", '{"expression": "4+4"}', call_id="c4"),
        text_turn("完"),
    ])
    loop = _loop_with_detect(client, window=3)
    events = await _collect(loop, "算")
    assert isinstance(events[-1], RunFinished)
    assert sum(isinstance(e, ToolFinished) for e in events) == 4


async def test_a_changed_result_resets_the_streak(make_mock, tool_turn, text_turn):
    """中间有一次结果变了就重新计数 —— 「连续」必须真的是连续。

    没有这一条，一次长会话里零散出现的相同结果会被攒起来误判。
    """
    client = make_mock([
        tool_turn("calculator", '{"expression": "1+1"}', call_id="c1"),
        tool_turn("calculator", '{"expression": "0+2"}', call_id="c2"),
        tool_turn("calculator", '{"expression": "9+9"}', call_id="c3"),  # 结果变了
        tool_turn("calculator", '{"expression": "4-2"}', call_id="c4"),
        tool_turn("calculator", '{"expression": "6-4"}', call_id="c5"),
        text_turn("完"),
    ])
    loop = _loop_with_detect(client, window=3)
    events = await _collect(loop, "算")
    assert isinstance(events[-1], RunFinished)
    assert sum(isinstance(e, ToolFinished) for e in events) == 5


async def test_result_stagnation_is_off_when_detection_is_off(
        make_mock, tool_turn, text_turn):
    """window<2 关掉整套检测，这一条也跟着关 —— 一个开关管一件事。"""
    client = make_mock(_same_result_turns(tool_turn, 4) + [text_turn("完")])
    loop = _loop_with_detect(client, window=0)
    events = await _collect(loop, "算")
    assert isinstance(events[-1], RunFinished)
