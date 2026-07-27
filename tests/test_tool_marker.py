"""ToolOutput.marker：机读标记进事件/落库，但不进模型上下文。

动机：save_download/save_to_knowledge 在结果尾部带〔下载ID:x〕〔知识ID:x〕供前端渲染
下载按钮、交付门清理产物。以前这串 id 是拼在结果正文里的，模型看见就当成有用信息抄进
回复（「知识库ID：ba87f8b5…」）——对用户是一串毫无意义的乱码。
"""
import pytest
from pydantic import BaseModel

from harness.context.manager import ContextManager
from harness.llm.base import StreamChunk, ToolCallDelta
from harness.loop.agent_loop import AgentLoop
from harness.tools.base import Tool, ToolExecutor, ToolRegistry
from harness.types import Role, ToolCall, ToolOutput


class _MarkedTool(Tool):
    name = "marked"
    description = "returns text plus a machine-readable marker"

    class Params(BaseModel):
        pass

    async def run(self, params):
        return ToolOutput(text="已保存到下载区：note.md。", marker="〔下载ID:abc123〕")


class _CapturingClient:
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


async def test_executor_splits_marker_from_model_content():
    reg = ToolRegistry(); reg.register(_MarkedTool())
    res = await ToolExecutor(reg).execute(ToolCall(id="c1", name="marked", arguments={}))
    assert res.content == "已保存到下载区：note.md。〔下载ID:abc123〕"   # 事件/落库/前端这份带标记
    assert res.for_model() == "已保存到下载区：note.md。"                # 喂模型这份不带


async def test_marker_survives_truncation():
    """标记接在截断之后：否则长结果会把它连同正文一起截掉，前端就渲染不出下载按钮。"""
    reg = ToolRegistry(); reg.register(_MarkedTool())
    res = await ToolExecutor(reg, max_chars=5).execute(
        ToolCall(id="c1", name="marked", arguments={}))
    assert res.content.endswith("〔下载ID:abc123〕")
    assert "…(已截断)" in res.content


async def test_plain_str_tool_unchanged():
    from harness.tools.builtins.calculator import CalculatorTool
    reg = ToolRegistry(); reg.register(CalculatorTool())
    res = await ToolExecutor(reg).execute(
        ToolCall(id="c1", name="calculator", arguments={"expression": "1+1"}))
    assert res.content == "2" and res.for_model() == "2"


async def test_loop_keeps_marker_out_of_model_context():
    reg = ToolRegistry(); reg.register(_MarkedTool())
    client = _CapturingClient([
        [StreamChunk(type="tool_call",
                     tool_call_delta=ToolCallDelta(index=0, id="c1", name="marked",
                                                   arguments="{}")),
         StreamChunk(type="done")],
        [StreamChunk(type="text", text="好了"), StreamChunk(type="done")],
    ])
    loop = AgentLoop(client=client, registry=reg, context=ContextManager(system_prompt="s"),
                     max_steps=5, run_id_factory=lambda: "r")
    [ev async for ev in loop.run("存成文件")]
    tool_msgs = [m for m in client.seen[1] if m.role == Role.TOOL]
    assert len(tool_msgs) == 1
    assert "abc123" not in tool_msgs[0].content      # 模型压根看不到这串 id
    assert "已保存到下载区" in tool_msgs[0].content   # 但知道存成功了
