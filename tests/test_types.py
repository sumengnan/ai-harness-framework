import json
from harness.types import Role, ToolCall, ToolResult, Message


def test_user_message_to_openai():
    msg = Message(role=Role.USER, content="hi")
    assert msg.to_openai() == {"role": "user", "content": "hi"}


def test_assistant_with_tool_calls_to_openai():
    msg = Message(
        role=Role.ASSISTANT,
        content=None,
        tool_calls=[ToolCall(id="c1", name="calculator", arguments={"expression": "1+1"})],
    )
    out = msg.to_openai()
    assert out["role"] == "assistant"
    assert out["tool_calls"][0]["id"] == "c1"
    assert out["tool_calls"][0]["type"] == "function"
    assert out["tool_calls"][0]["function"]["name"] == "calculator"
    assert json.loads(out["tool_calls"][0]["function"]["arguments"]) == {"expression": "1+1"}


def test_tool_result_message_to_openai():
    msg = Message(role=Role.TOOL, content="2", tool_call_id="c1")
    assert msg.to_openai() == {"role": "tool", "tool_call_id": "c1", "content": "2"}
