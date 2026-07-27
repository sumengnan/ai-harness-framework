# src/harness/approval.py
"""危险命令的人工确认回传通道。

工具的 run() 在检出危险命令后需要「阻塞等待人工批准」。但决策来自另一个 HTTP 请求
（POST /decision），与被阻塞的工具协程运行在同一事件循环但不同的 task/请求上下文中，
无法用 per-request contextvar 桥接。因此用一个**模块级全局 registry**，按全局唯一的
approval_id（uuid）索引一个 asyncio.Future——工具 await 该 Future，端点 resolve 它。

另用一个 contextvar 携带「本次运行的 run_id + 超时」。未设置上下文（纯 harness / CLI /
测试场景）时 request_approval 直接放行，保持内核对无人工通道的场景透明可用。

与 progress.emit 配合：request_approval 先 emit(ApprovalRequired) 再 await，事件经
emitter 并入 SSE 队列被前端立即消费；前端 POST 决策后 resolve() 令 Future 完成、工具恢复。
"""
from __future__ import annotations

import asyncio
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field

from .events import ApprovalRequired, ApprovalResolved
from .progress import emit

_pending: dict[str, asyncio.Future] = {}   # 跨请求：approval_id -> Future[bool]
_ctx: ContextVar = ContextVar("approval_ctx", default=None)   # ApprovalContext | None


@dataclass
class ApprovalContext:
    run_id: str
    timeout: float
    # 本轮已被拒绝的 (工具, 命令)。用户拒绝是**人做出的决定**，同一条命令再问一遍不会有
    # 不同答案，只会把弹窗怼到用户脸上第二次、第三次（编排器的单步重试正是这么干的）。
    # 故记下来，后续同样的请求直接拒、不再打扰。
    denied: set = field(default_factory=set)


def set_context(run_id: str, timeout: float):
    """设置本次运行的审批上下文，返回 token 供 reset。"""
    return _ctx.set(ApprovalContext(run_id, timeout))


def reset_context(token) -> None:
    _ctx.reset(token)


def set_run_id(run_id: str) -> None:
    """在 run_id 已知后回填到当前审批上下文（供 ApprovalRequired 事件自洽携带）。"""
    ctx = _ctx.get()
    if ctx is not None:
        ctx.run_id = run_id


async def request_approval(tool: str, command: str, reason: str) -> bool:
    """请求人工确认。无审批上下文（CLI/测试）→ 放行返回 True；超时/拒绝 → False。"""
    ctx = _ctx.get()
    if ctx is None:                       # 无人工通道 → 放行，保持内核纯净可用
        return True
    key = (tool, command)
    if key in ctx.denied:                 # 本轮已拒过同一条命令 → 直接拒，不再弹第二次
        return False
    approval_id = uuid.uuid4().hex
    fut: asyncio.Future = asyncio.get_event_loop().create_future()
    _pending[approval_id] = fut
    emit(ApprovalRequired(run_id=ctx.run_id, approval_id=approval_id,
                          tool=tool, command=command, reason=reason))   # 先发事件，再阻塞
    try:
        approved = await asyncio.wait_for(fut, ctx.timeout)
    except asyncio.TimeoutError:
        ctx.denied.add(key)
        emit(ApprovalResolved(approval_id=approval_id, approved=False, reason="超时未确认"))
        return False                      # 超时自动拒绝
    else:
        if not approved:
            ctx.denied.add(key)
        emit(ApprovalResolved(approval_id=approval_id, approved=approved))
        return approved
    finally:
        _pending.pop(approval_id, None)


def resolve(approval_id: str, approved: bool) -> bool:
    """端点回调：命中有效 pending 则完成其 Future 并返回 True；否则 False。"""
    fut = _pending.get(approval_id)
    if fut is None or fut.done():
        return False
    fut.set_result(approved)
    return True
