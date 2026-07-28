# src/harness/persistence/serialize.py
from __future__ import annotations

from ..events import (
    ApprovalRequired, ApprovalResolved, ModelUsage, Progress, ReasoningDelta, RunError,
    RunFinished, RunStarted, StepFinished, StepStarted, TextDelta, ToolCallRequested,
    ToolFinished, ToolStarted,
)
from ..state import RunState
from ..types import Message, Role, ToolCall


def toolcall_to_dict(tc: ToolCall) -> dict:
    return {"id": tc.id, "name": tc.name, "arguments": tc.arguments}


def toolcall_from_dict(d: dict) -> ToolCall:
    return ToolCall(id=d["id"], name=d["name"], arguments=d["arguments"])


def message_to_dict(m: Message) -> dict:
    return {
        "role": m.role.value,
        "content": m.content,
        "tool_calls": [toolcall_to_dict(tc) for tc in m.tool_calls],
        "tool_call_id": m.tool_call_id,
    }


def message_from_dict(d: dict) -> Message:
    return Message(
        role=Role(d["role"]),
        content=d.get("content"),
        tool_calls=[toolcall_from_dict(x) for x in d.get("tool_calls", [])],
        tool_call_id=d.get("tool_call_id"),
    )


def runstate_to_dict(s: RunState) -> dict:
    return {"run_id": s.run_id, "step": s.step,
            "messages": [message_to_dict(m) for m in s.messages],
            "tokens_used": s.tokens_used,
            "wall_seconds_used": s.wall_seconds_used}


def runstate_from_dict(d: dict) -> RunState:
    st = RunState(run_id=d["run_id"])
    st.step = d["step"]
    st.messages = [message_from_dict(m) for m in d["messages"]]
    # .get：这两个字段是后加的，老快照没有，按 0 读回来而不是崩
    st.tokens_used = d.get("tokens_used", 0)
    st.wall_seconds_used = d.get("wall_seconds_used", 0.0)
    return st


def event_to_dict(ev) -> dict:
    """单向：把事件序列化成 {type, data} 供轨迹存储/阅读。"""
    if isinstance(ev, RunStarted):
        data = {"run_id": ev.run_id}
    elif isinstance(ev, (StepStarted, StepFinished)):
        data = {"step": ev.step}
    elif isinstance(ev, ReasoningDelta):
        data = {"text": ev.text}
    elif isinstance(ev, TextDelta):
        data = {"text": ev.text}
    elif isinstance(ev, ToolCallRequested):
        data = {"tool_calls": [toolcall_to_dict(tc) for tc in ev.tool_calls]}
    elif isinstance(ev, ToolStarted):
        data = {"tool_call": toolcall_to_dict(ev.tool_call)}
    elif isinstance(ev, ToolFinished):
        r = ev.result
        data = {"result": {"tool_call_id": r.tool_call_id, "content": r.content,
                           "is_error": r.is_error, "meta": r.meta}}
    elif isinstance(ev, RunFinished):
        data = {"message": message_to_dict(ev.message)}
    elif isinstance(ev, RunError):
        data = {"error": ev.error}
    elif isinstance(ev, Progress):
        data = {"scope": ev.scope, "text": ev.text, "status": ev.status, "key": ev.key,
                "agent": ev.agent, "detail": ev.detail}
    elif isinstance(ev, ApprovalRequired):
        data = {"run_id": ev.run_id, "approval_id": ev.approval_id,
                "tool": ev.tool, "command": ev.command, "reason": ev.reason}
    elif isinstance(ev, ApprovalResolved):
        data = {"approval_id": ev.approval_id, "approved": ev.approved, "reason": ev.reason}
    elif isinstance(ev, ModelUsage):
        u = ev.usage
        data = {"usage": {"prompt": u.prompt_tokens, "completion": u.completion_tokens, "total": u.total_tokens},
                "cost_usd": ev.cost_usd, "attempts": ev.attempts, "latency_ms": ev.latency_ms,
                "model": ev.model}
    else:
        data = {}
    return {"type": type(ev).__name__, "data": data}
