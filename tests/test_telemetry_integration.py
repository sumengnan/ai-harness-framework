import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from harness.loop.agent_loop import AgentLoop
from harness.context.manager import ContextManager
from harness.tools.base import ToolRegistry
from harness.tools.builtins.calculator import CalculatorTool
from harness.llm.base import StreamChunk, ToolCallDelta
from harness.usage import Usage


def _tracer_and_exporter():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test"), exporter


async def test_run_produces_span_tree(make_mock):
    tracer, exporter = _tracer_and_exporter()
    tool_turn = [
        StreamChunk(type="tool_call", tool_call_delta=ToolCallDelta(
            index=0, id="c1", name="calculator", arguments='{"expression":"1+1"}')),
        StreamChunk(type="done", usage=Usage(10, 5, 15), attempts=1),
    ]
    text_turn = [StreamChunk(type="text", text="答案 2"), StreamChunk(type="done", usage=Usage(3, 2, 5))]
    reg = ToolRegistry()
    reg.register(CalculatorTool())
    loop = AgentLoop(client=make_mock([tool_turn, text_turn]), registry=reg,
                     context=ContextManager(system_prompt="s"), max_steps=5,
                     run_id_factory=lambda: "r1", tracer=tracer)
    _ = [e async for e in loop.run("算 1+1")]

    names = [s.name for s in exporter.get_finished_spans()]
    assert "run" in names
    assert "step" in names
    assert "model_call" in names
    assert "tool_call:calculator" in names


async def test_tool_error_marks_span_error(make_mock):
    from opentelemetry.trace import StatusCode

    tracer, exporter = _tracer_and_exporter()
    bad_tool_turn = [
        StreamChunk(type="tool_call", tool_call_delta=ToolCallDelta(
            index=0, id="c1", name="calculator", arguments="not-json")),
        StreamChunk(type="done", usage=Usage(10, 5, 15)),
    ]
    text_turn = [StreamChunk(type="text", text="抱歉，我重发"),
                 StreamChunk(type="done", usage=Usage(3, 2, 5))]
    reg = ToolRegistry()
    reg.register(CalculatorTool())
    loop = AgentLoop(client=make_mock([bad_tool_turn, text_turn]), registry=reg,
                     context=ContextManager(system_prompt="s"), max_steps=5,
                     run_id_factory=lambda: "r1", tracer=tracer)
    _ = [e async for e in loop.run("算点啥")]

    tool_spans = [s for s in exporter.get_finished_spans() if s.name.startswith("tool_call:")]
    assert tool_spans, "非法 JSON 分支也应在 tool_call span 内"
    assert any(s.status.status_code == StatusCode.ERROR for s in tool_spans)
    assert any(any(ev.name == "tool.error" for ev in s.events) for s in tool_spans)


async def test_retry_records_span_event(flaky_client, text_turn):
    from opentelemetry.trace import StatusCode
    from harness.reliability.retry import RetryingModelClient

    class Transient(Exception):
        pass

    async def fake_sleep(d):
        pass

    tracer, exporter = _tracer_and_exporter()
    inner = flaky_client(Transient("timeout"), text_turn("ok"), fail_times=1)
    retrying = RetryingModelClient(inner, max_retries=2, base_delay=0.1,
                                   sleep=fake_sleep, transient=(Transient,))
    reg = ToolRegistry()
    reg.register(CalculatorTool())
    loop = AgentLoop(client=retrying, registry=reg,
                     context=ContextManager(system_prompt="s"), max_steps=5,
                     run_id_factory=lambda: "r1", tracer=tracer)
    _ = [e async for e in loop.run("hi")]

    mc_spans = [s for s in exporter.get_finished_spans() if s.name == "model_call"]
    assert mc_spans
    assert any(any(ev.name == "model_call.retry" for ev in s.events) for s in mc_spans)
