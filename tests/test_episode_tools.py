# tests/test_episode_tools.py
from harness.memory.episodic import EpisodicMemory
from harness.memory.memory import Memory
from harness.memory.sqlite_backend import SqliteVecBackend
from harness.tools.base import ToolRegistry, ToolExecutor
from harness.tools.builtins.episode_tools import RecallEpisodesTool, RecordEpisodeTool
from harness.types import ToolCall


def _ep(mock_embedder):
    store = SqliteVecBackend(":memory:", dimension=64)
    mem = Memory(store, mock_embedder(dimension=64), chunk_size=1000, overlap=0)
    return EpisodicMemory(mem)


async def test_record_then_recall_tools(mock_embedder):
    ep = _ep(mock_embedder)
    reg = ToolRegistry()
    reg.register(RecordEpisodeTool(ep))
    reg.register(RecallEpisodesTool(ep, default_k=3))
    ex = ToolExecutor(reg)

    w = await ex.execute(ToolCall(id="c1", name="record_episode",
                                  arguments={"task": "排序数组", "lesson": "用快排最稳", "success": True}))
    assert w.is_error is False and "已记录" in w.content

    r = await ex.execute(ToolCall(id="c2", name="recall_episodes",
                                  arguments={"query": "排序"}))
    assert r.is_error is False and "快排" in r.content


async def test_recall_empty_message(mock_embedder):
    ep = _ep(mock_embedder)
    reg = ToolRegistry(); reg.register(RecallEpisodesTool(ep))
    ex = ToolExecutor(reg)
    r = await ex.execute(ToolCall(id="c1", name="recall_episodes", arguments={"query": "任何"}))
    assert "无相关历史经验" in r.content
