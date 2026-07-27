from harness.loop.agent_loop import AgentLoop
from harness.context.manager import ContextManager
from harness.tools.base import ToolRegistry
from harness.orchestration.spec import AgentSpec, AgentRoster
from harness.orchestration.dispatch import DispatchTool
from harness.tools.builtins.calculator import CalculatorTool
from harness.llm.base import StreamChunk, ToolCallDelta
from harness.events import ToolFinished, RunFinished
from harness.reliability.budget import BudgetTracker
from harness.usage import Usage


async def test_main_dispatches_and_summarizes(make_mock):
    dispatch_turn = [
        StreamChunk(type="tool_call", tool_call_delta=ToolCallDelta(
            index=0, id="c1", name="dispatch",
            arguments='{"agent": "researcher", "task": "算 6*7"}')),
        StreamChunk(type="done", usage=Usage(10, 5, 15)),
    ]
    sub_turn = [StreamChunk(type="text", text="42"),
                StreamChunk(type="done", usage=Usage(8, 2, 10))]
    final_turn = [StreamChunk(type="text", text="研究员算出 42"),
                  StreamChunk(type="done", usage=Usage(4, 3, 7))]
    client = make_mock([dispatch_turn, sub_turn, final_turn])

    roster = AgentRoster([AgentSpec("researcher", "研究员", "你是研究员", ["calculator"])])
    pool = {"calculator": CalculatorTool()}
    budget = BudgetTracker()
    dispatch = DispatchTool(roster, pool, client=client, budget=budget,
                            depth=0, max_depth=2, sub_max_steps=5)
    reg = ToolRegistry(); reg.register(dispatch)
    loop = AgentLoop(client=client, registry=reg,
                     context=ContextManager(system_prompt="主"), max_steps=5,
                     run_id_factory=lambda: "r1", budget=budget)

    events = [e async for e in loop.run("研究 6*7")]

    finished = [e for e in events if isinstance(e, ToolFinished)]
    assert finished[0].result.content == "42"            # 子 agent 独立上下文跑出的结果
    assert finished[0].result.is_error is False
    assert isinstance(events[-1], RunFinished)
    assert "42" in events[-1].message.content            # 主 agent 汇总
    assert budget.total_tokens == 32                     # 15+10+7 全树共享累计
