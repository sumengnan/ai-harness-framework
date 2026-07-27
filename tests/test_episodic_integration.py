# tests/test_episodic_integration.py
from harness.loop.agent_loop import AgentLoop
from harness.context.manager import ContextManager
from harness.tools.base import ToolRegistry
from harness.memory.episodic import EpisodicMemory, EpisodeRecorder
from harness.memory.memory import Memory
from harness.memory.sqlite_backend import SqliteVecBackend
from harness.tools.builtins.episode_tools import RecallEpisodesTool
from harness.llm.base import StreamChunk, ToolCallDelta
from harness.events import ToolFinished


async def test_recorded_run_recalled_next(make_mock, text_turn, mock_embedder):
    store = SqliteVecBackend(":memory:", dimension=64)
    mem = Memory(store, mock_embedder(dimension=64), chunk_size=1000, overlap=0)
    ep = EpisodicMemory(mem)
    rec = EpisodeRecorder(ep)

    # run1：无工具、直接作答；recorder 包裹 → 跑完自动记 episode
    loop1 = AgentLoop(client=make_mock([text_turn("用二分查找解决了")]),
                      registry=ToolRegistry(), context=ContextManager("s"),
                      max_steps=5, run_id_factory=lambda: "r1")
    _ = [e async for e in rec.wrap(loop1.run("在有序数组里查找"), task="在有序数组里查找")]

    # run2：recall_episodes 工具取到 run1 的经验
    reg = ToolRegistry(); reg.register(RecallEpisodesTool(ep, default_k=3))
    recall_turn = [
        StreamChunk(type="tool_call", tool_call_delta=ToolCallDelta(
            index=0, id="c1", name="recall_episodes", arguments='{"query": "查找"}')),
        StreamChunk(type="done"),
    ]
    loop2 = AgentLoop(client=make_mock([recall_turn, text_turn("参考历史，用二分")]),
                      registry=reg, context=ContextManager("s"),
                      max_steps=5, run_id_factory=lambda: "r2")
    events = [e async for e in loop2.run("怎么在有序数组查找？")]

    finished = [e for e in events if isinstance(e, ToolFinished)]
    assert "二分查找" in finished[0].result.content    # 召回了 run1 的经验
