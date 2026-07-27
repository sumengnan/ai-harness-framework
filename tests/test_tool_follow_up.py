import pytest
from pydantic import BaseModel

from harness.context.manager import ContextManager
from harness.llm.base import StreamChunk, ToolCallDelta
from harness.loop.agent_loop import AgentLoop
from harness.sandbox.local import LocalSandbox
from harness.tools.base import Tool, ToolExecutor, ToolRegistry
from harness.types import Message, Role, ToolCall, ToolOutput


class _VisionTool(Tool):
    name = "vision"
    description = "returns a follow-up image user message"

    class Params(BaseModel):
        pass

    async def run(self, params):
        follow = Message(role=Role.USER, content=[
            {"type": "text", "text": "img"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ])
        return ToolOutput(text="loaded", follow_up=[follow])


class _CapturingClient:
    """记录每次 stream() 收到的 messages，脚本化 yield 各轮 chunk。"""

    def __init__(self, turns):
        self._turns = list(turns)
        self._i = 0
        self.seen: list[list] = []

    async def stream(self, messages, tools):
        self.seen.append(list(messages))
        turn = self._turns[self._i]
        self._i += 1
        for chunk in turn:
            yield chunk


def _tool_turn(name, call_id="c1"):
    return [
        StreamChunk(type="tool_call",
                    tool_call_delta=ToolCallDelta(index=0, id=call_id, name=name, arguments="{}")),
        StreamChunk(type="done"),
    ]


def _text_turn(text):
    return [StreamChunk(type="text", text=text), StreamChunk(type="done")]


async def test_tool_executor_carries_follow_up():
    reg = ToolRegistry()
    reg.register(_VisionTool())
    ex = ToolExecutor(reg)
    res = await ex.execute(ToolCall(id="c1", name="vision", arguments={}))
    assert res.content == "loaded"
    assert len(res.follow_up) == 1
    assert res.follow_up[0].role == Role.USER
    assert isinstance(res.follow_up[0].content, list)


async def test_str_tool_has_empty_follow_up():
    from harness.tools.builtins.calculator import CalculatorTool
    reg = ToolRegistry(); reg.register(CalculatorTool())
    ex = ToolExecutor(reg)
    res = await ex.execute(ToolCall(id="c1", name="calculator", arguments={"expression": "1+1"}))
    assert res.content == "2" and res.follow_up == []


async def test_loop_injects_follow_up_into_next_model_call():
    reg = ToolRegistry(); reg.register(_VisionTool())
    client = _CapturingClient([_tool_turn("vision"), _text_turn("done")])
    loop = AgentLoop(client=client, registry=reg, context=ContextManager(system_prompt="s"),
                     max_steps=5, run_id_factory=lambda: "r")
    events = [ev async for ev in loop.run("看图")]
    # 第二次模型调用应能看到工具后注入的图片 user 消息
    second = client.seen[1]
    injected = [m for m in second if m.role == Role.USER and isinstance(m.content, list)]
    assert len(injected) == 1
    assert any(p.get("type") == "image_url" for p in injected[0].content)
    # 顺序：assistant(tool_calls) → tool → user(image)
    roles = [m.role for m in second]
    assert roles[-3:] == [Role.ASSISTANT, Role.TOOL, Role.USER]


async def test_local_sandbox_write_bytes_with_subdir(tmp_path):
    sb = LocalSandbox()
    await sb.start()
    try:
        await sb.write_bytes("uploads/pic.png", b"\x89PNG\x00\x01")
        import os
        p = os.path.join(sb.workspace, "uploads", "pic.png")
        assert os.path.exists(p)
        with open(p, "rb") as f:
            assert f.read() == b"\x89PNG\x00\x01"
    finally:
        await sb.close()
