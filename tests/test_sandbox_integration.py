from harness.loop.agent_loop import AgentLoop
from harness.context.manager import ContextManager
from harness.tools.base import ToolRegistry
from harness.sandbox.local import LocalSandbox
from harness.tools.builtins.code_tool import RunPythonTool
from harness.tools.builtins.fs_tools import WriteFileTool
from harness.llm.base import StreamChunk, ToolCallDelta
from harness.events import ToolFinished, RunFinished


async def test_agent_runs_code_in_sandbox(make_mock, text_turn):
    sb = LocalSandbox()
    await sb.start()
    try:
        reg = ToolRegistry()
        reg.register(RunPythonTool(sb, timeout=5))
        reg.register(WriteFileTool(sb))
        code_turn = [
            StreamChunk(type="tool_call", tool_call_delta=ToolCallDelta(
                index=0, id="c1", name="run_python",
                arguments='{"code": "print(6*7)"}')),
            StreamChunk(type="done"),
        ]
        loop = AgentLoop(client=make_mock([code_turn, text_turn("答案是 42")]),
                         registry=reg, context=ContextManager(system_prompt="s"),
                         max_steps=5, run_id_factory=lambda: "r1")
        events = [e async for e in loop.run("算 6*7")]
        finished = [e for e in events if isinstance(e, ToolFinished)]
        assert "42" in finished[0].result.content
        assert finished[0].result.is_error is False
        assert isinstance(events[-1], RunFinished)
    finally:
        await sb.close()
