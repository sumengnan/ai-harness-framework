# src/harness/progress.py
"""进度旁路：让工具（沙箱初始化、子 agent 派发等）在执行过程中向外发事件。

工具的 run() 只能返回字符串、无法 yield 事件；这里用 contextvar 存一个 emitter，
调用方（如 chat 路由）在进入 loop 前设置它、并把事件并入 SSE 流。默认 emitter 为空，
不设置时 emit() 无副作用（对纯 harness 使用透明）。
"""
from __future__ import annotations

import contextvars

_emitter: contextvars.ContextVar = contextvars.ContextVar("progress_emitter", default=None)
# 当前正在执行的子 agent 名（dispatch 期间设置）。沙箱等深层工具无从得知自己跑在哪个子
# agent 下，emit() 据此给 scope="sandbox" 的进度自动打上 agent 归属，便于前端区分。
_current_agent: contextvars.ContextVar = contextvars.ContextVar("current_agent", default=None)


def set_emitter(fn):
    """设置进度 emitter（fn 接收一个事件对象），返回 token 供 reset。"""
    return _emitter.set(fn)


def reset_emitter(token) -> None:
    _emitter.reset(token)


def set_current_agent(name):
    """标记后续进度归属的子 agent；返回 token 供 reset。"""
    return _current_agent.set(name)


def reset_current_agent(token) -> None:
    _current_agent.reset(token)


def emit(event) -> None:
    fn = _emitter.get()
    if fn is None:
        return
    # 子 agent 执行期间，给未标注归属的沙箱进度自动打标（不覆盖已有 agent）
    agent = _current_agent.get()
    if agent and getattr(event, "scope", None) == "sandbox" and getattr(event, "agent", None) is None:
        try:
            event.agent = agent
        except AttributeError:
            pass
    fn(event)
