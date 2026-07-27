import pytest
from harness.llm.base import StreamChunk


async def test_mock_yields_scripted_turns(make_mock, text_turn, tool_turn):
    client = make_mock([
        tool_turn("calculator", '{"expression": "1+1"}', call_id="c1"),
        text_turn("答案是 2"),
    ])
    # 第一轮：工具调用
    chunks = [c async for c in client.stream([], [])]
    assert any(c.type == "tool_call" for c in chunks)
    # 第二轮：纯文本
    chunks2 = [c async for c in client.stream([], [])]
    assert "".join(c.text for c in chunks2 if c.type == "text") == "答案是 2"
