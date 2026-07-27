# src/harness/loop/agent_loop.py
from __future__ import annotations

import contextlib
import json
import time
import uuid
from dataclasses import dataclass
from typing import AsyncIterator, Callable

from opentelemetry.trace import Status, StatusCode

from ..context.manager import ContextManager
from ..events import (
    Event, RunStarted, StepStarted, TextDelta, ReasoningDelta, ToolCallRequested,
    ToolStarted, ToolFinished, StepFinished, RunFinished, RunError, ModelUsage,
)
from ..llm.base import ModelClient, ToolCallDelta
from ..reliability.budget import BudgetTracker, BudgetExceeded
from ..state import RunState
from ..telemetry.tracer import get_tracer
from ..tools.base import ToolExecutor, ToolRegistry
from ..types import Message, Role, ToolCall, ToolResult
from ..usage import effective_cost


@dataclass
class _Finalized:
    call: ToolCall
    parse_error: str | None


def _accumulate(acc: dict[int, dict], delta: ToolCallDelta) -> None:
    slot = acc.setdefault(delta.index, {"id": None, "name": None, "args": ""})
    if delta.id:
        slot["id"] = delta.id
    if delta.name:
        slot["name"] = delta.name
    if delta.arguments:
        slot["args"] += delta.arguments


def _finalize(acc: dict[int, dict]) -> list[_Finalized]:
    out: list[_Finalized] = []
    for idx in sorted(acc):
        slot = acc[idx]
        parse_error: str | None = None
        args: dict = {}
        if slot["args"]:
            try:
                parsed = json.loads(slot["args"])
                if isinstance(parsed, dict):
                    args = parsed
                else:
                    parse_error = f"参数需为 JSON 对象，收到：{slot['args'][:80]}"
            except json.JSONDecodeError as e:
                parse_error = f"{e}：{slot['args'][:80]}"
        out.append(_Finalized(
            call=ToolCall(id=slot["id"] or f"call_{idx}", name=slot["name"] or "", arguments=args),
            parse_error=parse_error,
        ))
    return out


def _canonical_args(args) -> str:
    """把工具参数规范化成稳定字符串，供循环签名比较（键序/空白无关）。"""
    try:
        return json.dumps(args, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(args)


class AgentLoop:
    def __init__(
        self,
        client: ModelClient,
        registry: ToolRegistry,
        context: ContextManager,
        max_steps: int = 10,
        run_id_factory: Callable[[], str] | None = None,
        budget: BudgetTracker | None = None,
        tracer=None,
        model_name: str = "",
        price_map: dict | None = None,
        tool_result_max_chars: int | None = None,
        checkpoint_store=None,
        loop_detect_window: int = 0,
    ) -> None:
        self._client = client
        self._registry = registry
        self._executor = ToolExecutor(registry, max_chars=tool_result_max_chars)
        self._context = context
        self._max_steps = max_steps
        self._new_run_id = run_id_factory or (lambda: uuid.uuid4().hex)
        self._budget = budget
        self._tracer = tracer or get_tracer()
        self._model_name = model_name
        self._price_map = price_map or {}
        self._checkpoint_store = checkpoint_store
        # 循环/停滞检测窗口：连续 N 步相同工具调用签名即判打转并中止；<2 关闭。
        self._loop_window = loop_detect_window

    async def run(self, user_message: str | None = None, *,
                  messages: list[Message] | None = None) -> AsyncIterator[Event]:
        """启动一次 run。

        user_message：单条用户消息（旧用法，保持不变）。
        messages：完整的初始消息列表——调用方已经组装好上下文时使用
        （如把上一步的结构化诊断作为 assistant/user 轮次一并带入）。
        两者必须且只能给一个。
        """
        if (user_message is None) == (messages is None):
            raise ValueError("run() 需要 user_message 或 messages 之一，且不能同时提供")
        state = RunState(run_id=self._new_run_id())
        if messages is None:
            state.append(Message(role=Role.USER, content=user_message))
        else:
            for m in messages:
                state.append(m)
        async for ev in self._run_from(state, resuming=False):
            yield ev

    async def resume(self, run_id: str) -> AsyncIterator[Event]:
        """从 checkpoint 续跑 run_id。

        已知限制：从上一个完整步的下一步重跑；若中断发生在某步执行到一半，该步
        整步重跑，其中的**有副作用工具**（write_file / run_shell 等）可能被**重复
        执行**。调用方需保证工具幂等或自行去重。
        """
        if self._checkpoint_store is None:
            yield RunError(error="未配置 checkpoint_store，无法 resume")
            return
        state = self._checkpoint_store.load(run_id)
        if state is None:
            yield RunError(error=f"无 checkpoint：{run_id}")
            return
        async for ev in self._run_from(state, resuming=True):
            yield ev

    async def _run_from(self, state: RunState, resuming: bool) -> AsyncIterator[Event]:
        if self._budget:
            self._budget.start()
        # 回填 run_id 到审批上下文（若有），使 ApprovalRequired 事件自洽携带 run_id
        from ..approval import set_run_id
        set_run_id(state.run_id)
        if not resuming:
            yield RunStarted(run_id=state.run_id)

        recent_sigs: list = []          # 最近各步的工具调用签名，供循环检测
        loop_nudges = 0                 # 已注入纠偏次数；纠偏后仍循环即中止
        nudge_tmp_token = None          # 纠偏时的升温 token（见下方注入处），finally 必还原
        # ExitStack 兜住纠偏升温的还原：run() 有多个 return 出口（正常结束/循环中止/预算超限），
        # 逐个补 reset 迟早漏一个，而漏了就会把升温漏给调用方的上下文。
        with contextlib.ExitStack() as cleanup, \
                self._tracer.start_as_current_span("run") as run_span:
            run_span.set_attribute("harness.run_id", state.run_id)

            for step in range(state.step + 1, self._max_steps + 1):
                state.step = step

                if self._budget:  # 步边界预算检查
                    try:
                        self._budget.check()
                    except BudgetExceeded as e:
                        run_span.set_status(Status(StatusCode.ERROR, e.reason))
                        yield RunError(error=e.reason)
                        return

                yield StepStarted(step=step)

                with self._tracer.start_as_current_span("step") as step_span:
                    step_span.set_attribute("harness.step", step)

                    messages = self._context.build(state)
                    content_parts: list[str] = []
                    tool_acc: dict[int, dict] = {}
                    usage = None
                    attempts = 1
                    t0 = time.monotonic()
                    try:
                        with self._tracer.start_as_current_span("model_call") as mc_span:
                            mc_span.set_attribute("harness.model", self._model_name)
                            async for chunk in self._client.stream(messages, self._registry.schemas()):
                                if chunk.type == "text" and chunk.text:
                                    content_parts.append(chunk.text)
                                    yield TextDelta(text=chunk.text)
                                elif chunk.type == "reasoning" and chunk.text:
                                    yield ReasoningDelta(text=chunk.text)
                                elif chunk.type == "tool_call" and chunk.tool_call_delta:
                                    _accumulate(tool_acc, chunk.tool_call_delta)
                                elif chunk.type == "done":
                                    usage = chunk.usage
                                    attempts = chunk.attempts
                            if usage is not None:
                                mc_span.set_attribute("harness.tokens.total", usage.total_tokens)
                            mc_span.set_attribute("harness.attempts", attempts)
                    except Exception as e:
                        step_span.set_status(Status(StatusCode.ERROR, str(e)))
                        yield RunError(error=f"模型调用失败: {e}")
                        return

                    latency_ms = (time.monotonic() - t0) * 1000
                    if usage is not None:
                        cost = effective_cost(usage, self._model_name, self._price_map)
                        if self._budget:
                            self._budget.add_usage(usage)
                        yield ModelUsage(usage=usage, cost_usd=cost, attempts=attempts,
                                         latency_ms=latency_ms, model=self._model_name)

                    finalized = _finalize(tool_acc)
                    tool_calls = [f.call for f in finalized]
                    assistant = Message(
                        role=Role.ASSISTANT,
                        content="".join(content_parts) or None,
                        tool_calls=tool_calls,
                    )
                    state.append(assistant)

                    if not tool_calls:
                        yield RunFinished(message=assistant)
                        if self._checkpoint_store:
                            self._checkpoint_store.delete(state.run_id)
                        return

                    # 循环/停滞检测：连续 N 步发起完全相同的工具调用（同名+同参）判为原地打转。
                    # 先注入一次纠偏提示让模型换思路（不真执行本步重复调用，给 tool_calls 回填
                    # 「已跳过」结果以保持消息合法）；纠偏后仍继续重复才中止。window<2 关闭；
                    # 参数不同（如翻页）签名不同，不会误伤。
                    if self._loop_window >= 2:
                        sig = tuple(sorted(
                            (tc.name, _canonical_args(tc.arguments)) for tc in tool_calls))
                        recent_sigs.append(sig)
                        if (len(recent_sigs) >= self._loop_window
                                and len(set(recent_sigs[-self._loop_window:])) == 1):
                            names = "、".join(sorted({tc.name for tc in tool_calls}))
                            run_span.set_attribute("harness.loop_detected", True)
                            if loop_nudges < 1:      # 首次：注入纠偏，给模型换思路的机会
                                loop_nudges += 1
                                recent_sigs.clear()  # 重置窗口，让纠偏后的行为重新计数
                                # 连同采样一起换：只改措辞不改温度时，模型很容易照着同一条
                                # 采样路径再走一遍——「请换个思路」这句话本身也是低概率才
                                # 被听进去的。升温到本轮剩余步骤结束（finally 还原）。
                                from ..llm.sampling import (
                                    get_nudge_delta, pop_temperature_delta,
                                    push_temperature_delta)
                                if nudge_tmp_token is None and get_nudge_delta():
                                    nudge_tmp_token = push_temperature_delta(get_nudge_delta())
                                    cleanup.callback(pop_temperature_delta, nudge_tmp_token)
                                for tc in tool_calls:  # 回填跳过结果，保持 tool_calls 消息合法
                                    state.append(Message(
                                        role=Role.TOOL, tool_call_id=tc.id,
                                        content="检测到重复调用，已跳过本次执行。"))
                                state.append(Message(role=Role.USER, content=(
                                    f"系统提示：你已连续多次以相同参数调用「{names}」，疑似原地打转。"
                                    "请换个思路——改用不同的参数或工具，或如果掌握的信息已足够，"
                                    "就直接给出最终答复；不要再重复同样的调用。")))
                                yield StepFinished(step=step)
                                if self._checkpoint_store:
                                    self._checkpoint_store.save(state)
                                continue
                            # 纠偏后仍重复 → 中止
                            yield RunError(
                                error=f"检测到疑似循环：纠偏后仍连续重复相同的工具调用"
                                      f"（{names}），已中止")
                            return

                    yield ToolCallRequested(tool_calls=tool_calls)
                    for f in finalized:
                        tc = f.call
                        yield ToolStarted(tool_call=tc)
                        with self._tracer.start_as_current_span(f"tool_call:{tc.name}") as ts:
                            if f.parse_error:  # 自纠正：回填明确错误，让模型下一步重发
                                result = ToolResult(
                                    tc.id,
                                    f"工具调用参数不是合法 JSON：{f.parse_error}，请重新调用。",
                                    is_error=True,
                                )
                            else:
                                result = await self._executor.execute(tc)
                            ts.set_attribute("harness.tool.is_error", result.is_error)
                            if result.is_error:
                                ts.set_status(Status(StatusCode.ERROR, result.content[:200]))
                                ts.add_event("tool.error", {"content": result.content[:200]})
                        # for_model()：机读 marker（〔下载ID:x〕等）只进事件/落库，不进上下文
                        state.append(Message(role=Role.TOOL, content=result.for_model(),
                                             tool_call_id=tc.id))
                        # 工具追加的后续消息（如把图片作为 user 视觉块注入）：接在 tool 结果之后
                        for fm in result.follow_up:
                            state.append(fm)
                        yield ToolFinished(result=result)
                    yield StepFinished(step=step)
                    if self._checkpoint_store:  # 步边界存快照
                        self._checkpoint_store.save(state)

            yield RunError(error=f"达到 max_steps 上限 ({self._max_steps})")
