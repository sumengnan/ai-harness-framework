from harness.loop.agent_loop import AgentLoop
from harness.context.manager import ContextManager
from harness.tools.base import ToolRegistry
from harness.tools.builtins.calculator import CalculatorTool
from harness.persistence.checkpoint import CheckpointStore
from harness.state import RunState
from harness.types import Message, Role, ToolCall
from harness.llm.base import StreamChunk, ToolCallDelta
from harness.events import RunFinished, RunError, StepStarted, RunStarted


class _RecordingStore:
    def __init__(self): self.saves = []; self.deleted = []
    def save(self, state): self.saves.append(state.step)
    def load(self, run_id): return None
    def delete(self, run_id): self.deleted.append(run_id)


class _CapturingClient:
    """记录每次 stream 收到的 messages，供断言历史是否进上下文。"""

    def __init__(self, turns):
        self._turns = list(turns); self._i = 0; self.seen_messages = []

    async def stream(self, messages, tools):
        self.seen_messages.append(messages)
        turn = self._turns[self._i]; self._i += 1
        for chunk in turn:
            yield chunk


def _reg():
    r = ToolRegistry(); r.register(CalculatorTool()); return r


async def test_checkpoint_saved_each_step_deleted_on_finish(make_mock, text_turn):
    store = _RecordingStore()
    tool_turn = [StreamChunk(type="tool_call", tool_call_delta=ToolCallDelta(
        index=0, id="c1", name="calculator", arguments='{"expression":"1+1"}')),
        StreamChunk(type="done")]
    loop = AgentLoop(client=make_mock([tool_turn, text_turn("答案 2")]),
                     registry=_reg(), context=ContextManager("s"), max_steps=5,
                     run_id_factory=lambda: "r1", checkpoint_store=store)
    _ = [e async for e in loop.run("算 1+1")]
    assert store.saves == [1]              # step1 结束存一次
    assert store.deleted == ["r1"]         # RunFinished 删


async def test_backward_compat_no_store_same_events(make_mock, text_turn):
    loop = AgentLoop(client=make_mock([text_turn("hi")]), registry=ToolRegistry(),
                     context=ContextManager("s"), max_steps=5, run_id_factory=lambda: "r1")
    events = [e async for e in loop.run("hi")]
    assert isinstance(events[0], RunStarted) and isinstance(events[-1], RunFinished)


async def test_resume_continues_from_checkpoint(make_mock, text_turn):
    cs = CheckpointStore(":memory:")
    st = RunState(run_id="r1"); st.step = 1
    st.append(Message(role=Role.USER, content="算 1+1"))
    st.append(Message(role=Role.ASSISTANT, content=None,
                      tool_calls=[ToolCall(id="c1", name="calculator", arguments={"expression": "1+1"})]))
    st.append(Message(role=Role.TOOL, content="2", tool_call_id="c1"))
    cs.save(st)

    loop = AgentLoop(client=make_mock([text_turn("最终答案 2")]),
                     registry=_reg(), context=ContextManager("s"), max_steps=5,
                     checkpoint_store=cs)
    events = [e async for e in loop.resume("r1")]
    assert not any(isinstance(e, RunStarted) for e in events)          # resume 不再发 RunStarted
    assert [e.step for e in events if isinstance(e, StepStarted)] == [2]  # 从 step2 续
    assert isinstance(events[-1], RunFinished)
    assert "最终答案 2" in events[-1].message.content
    assert cs.load("r1") is None                                       # 完成后快照删


async def test_resume_missing_checkpoint_run_error(make_mock, text_turn):
    loop = AgentLoop(client=make_mock([text_turn("x")]), registry=ToolRegistry(),
                     context=ContextManager("s"), checkpoint_store=CheckpointStore(":memory:"))
    events = [e async for e in loop.resume("nope")]
    assert isinstance(events[-1], RunError) and "无 checkpoint" in events[-1].error


async def test_resume_feeds_history_into_context(text_turn):
    """resume 后模型收到的 messages 应含恢复的历史（验收标准4：恢复历史进上下文）。"""
    cs = CheckpointStore(":memory:")
    st = RunState(run_id="r1"); st.step = 1
    st.append(Message(role=Role.USER, content="算 1+1"))
    st.append(Message(role=Role.ASSISTANT, content=None,
                      tool_calls=[ToolCall(id="c1", name="calculator", arguments={"expression": "1+1"})]))
    st.append(Message(role=Role.TOOL, content="2", tool_call_id="c1"))
    cs.save(st)

    client = _CapturingClient([text_turn("最终答案 2")])
    loop = AgentLoop(client=client, registry=_reg(), context=ContextManager("s"),
                     max_steps=5, checkpoint_store=cs)
    _ = [e async for e in loop.resume("r1")]

    msgs = client.seen_messages[0]                          # resume 首次模型调用收到的上下文
    contents = [m.content for m in msgs]
    assert "算 1+1" in contents                             # 恢复的 user
    assert "2" in contents                                  # 恢复的 tool 结果
    # 恢复的 assistant tool_call 也在上下文
    assert any(m.tool_calls for m in msgs)


async def test_run_error_preserves_checkpoint(text_turn):
    """走到 RunError（max_steps 上限）时快照应保留，可再 resume。"""
    cs = CheckpointStore(":memory:")
    tool_turn = [StreamChunk(type="tool_call", tool_call_delta=ToolCallDelta(
        index=0, id="c1", name="calculator", arguments='{"expression":"1+1"}')),
        StreamChunk(type="done")]
    # max_steps=1：step1 是工具调用 → StepFinished 存快照 → 无 step2 → RunError 上限
    loop = AgentLoop(client=_CapturingClient([tool_turn]), registry=_reg(),
                     context=ContextManager("s"), max_steps=1,
                     run_id_factory=lambda: "r1", checkpoint_store=cs)
    events = [e async for e in loop.run("算 1+1")]
    assert isinstance(events[-1], RunError)
    assert cs.load("r1") is not None                        # RunError 保留快照
    assert cs.load("r1").step == 1


async def test_budget_carries_across_resume(text_turn_usage, done_with_usage):
    """预算跟着快照走：续跑接着上次的用量算，而不是重新给满一份。"""
    from harness.reliability.budget import BudgetTracker

    cs = CheckpointStore(":memory:")
    tool_turn = [StreamChunk(type="tool_call", tool_call_delta=ToolCallDelta(
        index=0, id="c1", name="calculator", arguments='{"expression":"1+1"}')),
        done_with_usage(prompt=40, completion=10)]        # 本步耗 50 tokens

    # max_steps=1：step1 存快照后即达上限 → RunError，快照保留
    loop = AgentLoop(client=_CapturingClient([tool_turn]), registry=_reg(),
                     context=ContextManager("s"), max_steps=1,
                     run_id_factory=lambda: "r1", checkpoint_store=cs,
                     budget=BudgetTracker(max_tokens=60))
    events = [e async for e in loop.run("算 1+1")]
    assert isinstance(events[-1], RunError)
    assert cs.load("r1").tokens_used == 50                # 用量进了快照

    # 续跑：新进程 → 新 BudgetTracker，但已用 50 会被装回来。
    # 预算在步边界检查，所以续跑的 step2 先跑完（累计 100），step3 开头才拦下。
    tool_turn2 = [StreamChunk(type="tool_call", tool_call_delta=ToolCallDelta(
        index=0, id="c2", name="calculator", arguments='{"expression":"2+2"}')),
        done_with_usage(prompt=40, completion=10)]
    loop2 = AgentLoop(client=_CapturingClient([tool_turn2, text_turn_usage("答案 2", 40, 10)]),
                      registry=_reg(), context=ContextManager("s"), max_steps=5,
                      checkpoint_store=cs, budget=BudgetTracker(max_tokens=60))
    ev2 = [e async for e in loop2.resume("r1")]
    assert isinstance(ev2[-1], RunError)
    assert "token 预算超限" in ev2[-1].error and "100 > 60" in ev2[-1].error

    # 对照：不装回来的话累计只有 50，同样的续跑不会被拦
    cs2 = CheckpointStore(":memory:")
    stale = cs.load("r1") or RunState(run_id="r1")
    stale.tokens_used = 0                                  # 模拟"预算没跟着快照走"
    cs2.save(stale)
    loop3 = AgentLoop(client=_CapturingClient([tool_turn2, text_turn_usage("答案 2", 40, 10)]),
                      registry=_reg(), context=ContextManager("s"), max_steps=5,
                      checkpoint_store=cs2, budget=BudgetTracker(max_tokens=60))
    ev3 = [e async for e in loop3.resume("r1")]
    assert isinstance(ev3[-1], RunFinished)                 # 50 < 60，跑完


async def test_budget_not_carried_without_snapshot_field(text_turn):
    """老快照没有 tokens_used 字段时按 0 读回，不崩。"""
    import json

    from harness.persistence.serialize import runstate_from_dict

    old = {"run_id": "r1", "step": 1, "messages": [
        {"role": "user", "content": "hi", "tool_calls": [], "tool_call_id": None}]}
    st = runstate_from_dict(json.loads(json.dumps(old)))
    assert st.tokens_used == 0 and st.wall_seconds_used == 0.0
    assert st.step == 1 and st.messages[0].content == "hi"
