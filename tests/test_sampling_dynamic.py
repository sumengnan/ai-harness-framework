"""动态增减温度：打转纠偏升温 + 还原不外泄。

策略值（升多少/降多少）在 tests/app/test_sampling_policy.py 里定；这里只管挂载点——
该升的时候真的升了，且不管从哪个出口离开都还原干净。
"""
from harness.context.manager import ContextManager
from harness.llm.sampling import (
    get_temperature_delta,
    reset_nudge_delta,
    resolve_sampling,
    set_nudge_delta,
)
from harness.loop.agent_loop import AgentLoop
from harness.tools.base import ToolRegistry
from harness.tools.builtins.calculator import CalculatorTool


def _loop(client, window=2, max_steps=10):
    reg = ToolRegistry(); reg.register(CalculatorTool())
    return AgentLoop(client=client, registry=reg,
                     context=ContextManager(system_prompt="s"),
                     max_steps=max_steps, run_id_factory=lambda: "run-test",
                     loop_detect_window=window)


class _TempSpy:
    """每次模型调用时记下当刻会发出的温度。"""
    def __init__(self, inner):
        self._inner = inner
        self.temps = []

    async def stream(self, messages, tools):
        self.temps.append(resolve_sampling(0.7).get("temperature"))
        async for c in self._inner.stream(messages, tools):
            yield c


async def test_nudge_raises_temperature_for_the_rest_of_the_run(
        make_mock, tool_turn, text_turn):
    """连续重复调用触发纠偏 → 之后的调用升温（只换措辞不换采样，模型多半照走老路）。"""
    spy = _TempSpy(make_mock([
        tool_turn("calculator", '{"expression": "1+1"}', call_id="c1"),
        tool_turn("calculator", '{"expression": "1+1"}', call_id="c1"),
        text_turn("好了"),
    ]))
    token = set_nudge_delta(0.2)
    try:
        _ = [ev async for ev in _loop(spy).run("算")]
    finally:
        reset_nudge_delta(token)
    assert spy.temps[0] == 0.7 and spy.temps[1] == 0.7    # 纠偏前照旧
    assert spy.temps[2] > spy.temps[1]                    # 纠偏那一步之后升温


async def test_nudge_delta_does_not_leak_out_of_the_run(make_mock, tool_turn, text_turn):
    """生成器不自带上下文隔离，升温漏出去会污染后续所有调用（含判分）。"""
    client = make_mock([
        tool_turn("calculator", '{"expression": "1+1"}', call_id="c1"),
        tool_turn("calculator", '{"expression": "1+1"}', call_id="c1"),
        text_turn("好了"),
    ])
    token = set_nudge_delta(0.2)
    try:
        _ = [ev async for ev in _loop(client).run("算")]
    finally:
        reset_nudge_delta(token)
    assert get_temperature_delta() == 0.0


async def test_no_leak_when_run_aborts_on_persistent_loop(make_mock, tool_turn):
    """纠偏后仍重复 → run 从中途 return 中止；这条出口也必须还原。"""
    client = make_mock([
        tool_turn("calculator", '{"expression": "1+1"}', call_id="c1"),
        tool_turn("calculator", '{"expression": "1+1"}', call_id="c1"),
        tool_turn("calculator", '{"expression": "1+1"}', call_id="c1"),
        tool_turn("calculator", '{"expression": "1+1"}', call_id="c1"),
    ])
    token = set_nudge_delta(0.2)
    try:
        _ = [ev async for ev in _loop(client).run("算")]
    finally:
        reset_nudge_delta(token)
    assert get_temperature_delta() == 0.0


async def test_nudge_disabled_when_delta_is_zero(make_mock, tool_turn, text_turn):
    """enable_dynamic_temperature=False 时 delta 为 0——纠偏照旧，只是不动温度。"""
    spy = _TempSpy(make_mock([
        tool_turn("calculator", '{"expression": "1+1"}', call_id="c1"),
        tool_turn("calculator", '{"expression": "1+1"}', call_id="c1"),
        text_turn("好了"),
    ]))
    _ = [ev async for ev in _loop(spy).run("算")]
    assert spy.temps == [0.7, 0.7, 0.7]
