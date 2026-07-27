from harness.loop.agent_loop import AgentLoop
from harness.context.manager import ContextManager
from harness.tools.base import ToolRegistry
from harness.tools.builtins.memory_search import SearchKnowledgeTool
from harness.memory.memory import Memory
from harness.memory.sqlite_backend import SqliteVecBackend
from harness.llm.base import StreamChunk, ToolCallDelta
from harness.events import ToolFinished, RunFinished


async def test_agent_uses_search_knowledge(make_mock, text_turn, mock_embedder):
    mem = Memory(SqliteVecBackend(":memory:", dimension=64),
                 mock_embedder(dimension=64), chunk_size=1000, overlap=0)
    await mem.add_texts(["光合作用把二氧化碳和水转化为葡萄糖和氧气"], "knowledge",
                        {"source": "生物笔记"})

    reg = ToolRegistry()
    reg.register(SearchKnowledgeTool(mem))
    ctx = ContextManager(system_prompt="s")

    search_turn = [
        StreamChunk(type="tool_call", tool_call_delta=ToolCallDelta(
            index=0, id="c1", name="search_knowledge",
            arguments='{"query": "光合作用", "k": 3}')),
        StreamChunk(type="done"),
    ]
    loop = AgentLoop(client=make_mock([search_turn, text_turn("据知识库，答案如上")]),
                     registry=reg, context=ctx, max_steps=5,
                     run_id_factory=lambda: "r1")
    events = [e async for e in loop.run("什么是光合作用？")]

    finished = [e for e in events if isinstance(e, ToolFinished)]
    assert "光合作用" in finished[0].result.content        # 检索结果回填
    assert finished[0].result.is_error is False
    assert isinstance(events[-1], RunFinished)
