from harness.loop.agent_loop import AgentLoop
from harness.context.manager import ContextManager
from harness.tools.base import ToolRegistry
from harness.persistence.trajectory import TrajectoryStore, TrajectorySink
from harness.persistence.checkpoint import CheckpointStore
from harness.state import RunState
from harness.types import Message, Role, ToolCall


async def test_real_loop_trajectory_persisted(make_mock, text_turn):
    store = TrajectoryStore(":memory:")
    sink = TrajectorySink(store)
    loop = AgentLoop(client=make_mock([text_turn("你好")]), registry=ToolRegistry(),
                     context=ContextManager("s"), max_steps=5, run_id_factory=lambda: "r1")
    _ = [e async for e in sink.wrap(loop.run("hi"))]
    types = [e["type"] for e in store.load("r1")]
    assert types[0] == "RunStarted"
    assert types[-1] == "RunFinished"
    assert "TextDelta" in types


async def test_resume_trajectory_appends_without_overwrite(make_mock, text_turn):
    """resume 段轨迹要续到原轨迹之后：既有原段又有恢复段，seq 连续、恢复段在后。"""
    store = TrajectoryStore(":memory:")
    sink = TrajectorySink(store)
    cs = CheckpointStore(":memory:")

    # 先跑一段普通 run，落轨迹（这段结束于 RunFinished，但我们随后手工存 checkpoint 模拟中断续跑）
    loop1 = AgentLoop(client=make_mock([text_turn("第一段")]), registry=ToolRegistry(),
                      context=ContextManager("s"), max_steps=5, run_id_factory=lambda: "r1")
    _ = [e async for e in sink.wrap(loop1.run("hi"))]
    first_len = len(store.load("r1"))
    assert first_len > 0

    # 手工存一个 checkpoint 供 resume
    st = RunState(run_id="r1"); st.step = 1
    st.append(Message(role=Role.USER, content="hi"))
    cs.save(st)

    loop2 = AgentLoop(client=make_mock([text_turn("恢复段")]), registry=ToolRegistry(),
                      context=ContextManager("s"), max_steps=5, checkpoint_store=cs)
    _ = [e async for e in sink.wrap(loop2.resume("r1"), run_id="r1")]

    all_events = store.load("r1")
    assert len(all_events) > first_len                       # 恢复段追加，未覆盖
    assert all_events[0]["type"] == "RunStarted"             # 原段仍在
    assert all_events[-1]["type"] == "RunFinished"           # 恢复段结尾在后
    # seq 连续无缺口
    seqs = [r[0] for r in store._conn.execute(
        "SELECT seq FROM trajectory_events WHERE run_id='r1' ORDER BY seq").fetchall()]
    assert seqs == list(range(len(seqs)))
