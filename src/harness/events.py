from __future__ import annotations

from dataclasses import dataclass, field

from .types import Message, ToolCall, ToolResult
from .usage import Usage


class Event:
    """所有事件的基类。"""


@dataclass
class RunStarted(Event):
    run_id: str


@dataclass
class StepStarted(Event):
    step: int


@dataclass
class TextDelta(Event):
    text: str


@dataclass
class ReasoningDelta(Event):
    """思考模式（如 Qwen3 reasoning_content）的推理增量。先于正文产出，
    用于让用户看到模型正在思考（也解释了首字为何较慢）。"""
    text: str


@dataclass
class ToolCallRequested(Event):
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class ToolStarted(Event):
    tool_call: ToolCall


@dataclass
class ToolFinished(Event):
    result: ToolResult


@dataclass
class StepFinished(Event):
    step: int


@dataclass
class RunFinished(Event):
    message: Message


@dataclass
class RunError(Event):
    error: str


@dataclass
class Progress(Event):
    """执行过程中的进度旁路事件：沙箱初始化、子 agent 派发进度等。"""
    scope: str   # "sandbox" | f"subagent:{agent}"
    text: str
    status: str | None = None   # "running" | "ok" | "error"，子 agent 每步的状态
    key: str | None = None      # 步骤合并键（工具调用 id）；前端据此把同一步的开始/完成折叠为一行
    agent: str | None = None    # 归属子 agent 名：沙箱步骤在子 agent 执行期间由 emit 自动打标
    detail: dict | None = None  # 工具调用明细（tool/args/result/is_error）供前端展开；普通进度行为 None


@dataclass
class ApprovalRequired(Event):
    """检出危险命令、需要人工确认时发出；前端据此弹窗，用 approval_id 回传决策。"""
    run_id: str
    approval_id: str
    tool: str
    command: str
    reason: str


@dataclass
class ApprovalResolved(Event):
    """审批已决（批准/拒绝/超时）；前端据此关闭弹窗。"""
    approval_id: str
    approved: bool
    reason: str | None = None


@dataclass
class ModelUsage(Event):
    usage: Usage
    cost_usd: float | None
    attempts: int
    latency_ms: float
    model: str | None = None   # 产生该用量的模型名；供按模型分组统计/计价。聚合快照可为 None
