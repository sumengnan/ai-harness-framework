# tests/test_episodic.py
from harness.memory.episodic import Episode, EpisodicMemory, EpisodeRecorder
from harness.memory.memory import Memory
from harness.memory.sqlite_backend import SqliteVecBackend
from harness.events import RunFinished, RunError, TextDelta
from harness.types import Message, Role


def _episodic(mock_embedder):
    store = SqliteVecBackend(":memory:", dimension=64)
    mem = Memory(store, mock_embedder(dimension=64), chunk_size=1000, overlap=0)
    return EpisodicMemory(mem)


async def _gen(evs):
    for e in evs:
        yield e


def test_episode_to_text():
    assert "成功" in Episode("t", "o", True).to_text()
    assert "失败" in Episode("t", "o", False).to_text()


async def test_record_and_recall(mock_embedder):
    ep = _episodic(mock_embedder)
    await ep.record("给数组排序", "用了快速排序，通过", True)
    hits = await ep.recall("排序", 3)
    assert any("快速排序" in h.text for h in hits)


async def test_recorder_records_success_on_finish(mock_embedder):
    ep = _episodic(mock_embedder)
    rec = EpisodeRecorder(ep)
    evs = [TextDelta(text="hi"),
           RunFinished(message=Message(role=Role.ASSISTANT, content="搞定了"))]
    out = [e async for e in rec.wrap(_gen(evs), task="做个任务")]
    assert len(out) == 2                              # 透传全部事件
    hits = await ep.recall("任务", 3)
    assert any("成功" in h.text for h in hits)


async def test_recorder_records_failure_on_error(mock_embedder):
    ep = _episodic(mock_embedder)
    rec = EpisodeRecorder(ep)
    _ = [e async for e in rec.wrap(_gen([RunError(error="超时了")]), task="失败任务")]
    hits = await ep.recall("失败任务", 3)
    assert any("失败" in h.text for h in hits)


async def test_recorder_records_even_when_consumer_breaks_after_terminal(mock_embedder):
    ep = _episodic(mock_embedder)
    rec = EpisodeRecorder(ep)
    gen = rec.wrap(_gen([RunFinished(message=Message(role=Role.ASSISTANT, content="done"))]), task="任务X")
    async for ev in gen:
        if isinstance(ev, RunFinished):
            break
    await gen.aclose()
    hits = await ep.recall("任务X", 3)
    assert any("成功" in h.text for h in hits)   # break-after-terminal 仍记录


async def test_recorder_no_terminal_no_record(mock_embedder):
    ep = _episodic(mock_embedder)
    rec = EpisodeRecorder(ep)
    _ = [e async for e in rec.wrap(_gen([TextDelta(text="only")]), task="半截任务")]
    assert await ep.recall("半截任务", 3) == []      # 无终止事件 → 不记录


async def test_episodes_isolated_from_knowledge(mock_embedder):
    store = SqliteVecBackend(":memory:", dimension=64)
    mem = Memory(store, mock_embedder(dimension=64), chunk_size=1000, overlap=0)
    ep = EpisodicMemory(mem)
    await mem.add_texts(["知识库内容"], "knowledge")
    await ep.record("任务A", "结果A", True)
    kn = await mem.search("内容", "knowledge", 5)
    assert all(h.collection == "knowledge" for h in kn)
    eh = await ep.recall("任务", 5)
    assert all(h.collection == "episodes" for h in eh)
