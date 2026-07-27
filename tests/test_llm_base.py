from harness.llm.base import StreamChunk, ToolCallDelta, ModelClient


def test_stream_chunk_text():
    c = StreamChunk(type="text", text="hello")
    assert c.type == "text"
    assert c.text == "hello"
    assert c.tool_call_delta is None


def test_stream_chunk_tool_call():
    d = ToolCallDelta(index=0, id="c1", name="calculator", arguments='{"exp')
    c = StreamChunk(type="tool_call", tool_call_delta=d)
    assert c.tool_call_delta.name == "calculator"
    assert c.tool_call_delta.arguments == '{"exp'


def test_model_client_is_protocol():
    # Protocol 的替代验证：接口方法存在
    assert hasattr(ModelClient, "stream")


def test_stream_chunk_done_carries_usage_and_attempts():
    from harness.usage import Usage
    c = StreamChunk(type="done", usage=Usage(1, 2, 3), attempts=2)
    assert c.usage.total_tokens == 3
    assert c.attempts == 2


def test_stream_chunk_defaults_attempts_one():
    assert StreamChunk(type="text", text="x").attempts == 1
