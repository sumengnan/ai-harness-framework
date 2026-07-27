from harness.events import (
    RunStarted, StepStarted, TextDelta, ToolCallRequested,
    ToolStarted, ToolFinished, StepFinished, RunFinished, RunError, Event,
)
from harness.types import Message, Role, ToolCall, ToolResult


def test_events_are_event_subclasses():
    assert isinstance(RunStarted(run_id="r1"), Event)
    assert isinstance(TextDelta(text="hi"), Event)


def test_event_payloads():
    assert TextDelta(text="hi").text == "hi"
    assert StepStarted(step=1).step == 1
    tc = ToolCall(id="c1", name="calculator", arguments={})
    assert ToolCallRequested(tool_calls=[tc]).tool_calls == [tc]
    tr = ToolResult(tool_call_id="c1", content="2")
    assert ToolFinished(result=tr).result is tr
    m = Message(role=Role.ASSISTANT, content="done")
    assert RunFinished(message=m).message is m
    assert RunError(error="boom").error == "boom"


def test_model_usage_event():
    from harness.events import ModelUsage
    from harness.usage import Usage
    ev = ModelUsage(usage=Usage(1, 2, 3), cost_usd=0.5, attempts=1, latency_ms=12.0)
    assert ev.usage.total_tokens == 3
    assert ev.cost_usd == 0.5
    assert ev.attempts == 1
    assert ev.latency_ms == 12.0
