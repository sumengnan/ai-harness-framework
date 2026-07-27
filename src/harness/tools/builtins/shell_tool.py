# src/harness/tools/builtins/shell_tool.py
from __future__ import annotations

from pydantic import BaseModel

from ..base import Tool, ToolError
from ...types import ToolOutput
from ...approval import request_approval
from ...events import Progress
from ...progress import emit
from ...sandbox.base import Sandbox
from ...shell.policy import classify_command
from ._sandbox_util import format_exec


class RunShellTool(Tool):
    name = "run_shell"
    description = "在沙箱内执行 shell 命令，返回 stdout/stderr/退出码。"

    class Params(BaseModel):
        command: str

    def __init__(self, sandbox: Sandbox, timeout: float = 30.0, max_chars: int = 8000) -> None:
        self._sandbox = sandbox
        self._timeout = timeout
        self._max_chars = max_chars

    async def run(self, params: "RunShellTool.Params") -> str:
        # 危险命令拦截：命中黑名单则请求人工确认，拒绝/超时 → 中止并回传错误给模型
        danger = classify_command(params.command)
        if danger is not None:
            approved = await request_approval("run_shell", params.command, danger.reason)
            if not approved:
                # 只陈述「被拒绝」不够：模型的默认倾向是把任务讲圆，实测会照着预期编出
                # 「目录已删除、当前已不存在」这类根本没发生的结果——在安全关键动作上撒谎。
                # 故这里必须连「该怎么向用户交代」一起说死。
                # 确定性可见记录：上面那段是给模型的约束，属软约束，管不住它撒谎。这条
                # 走 scope=check 进度（落库、刷新后仍在、前端红色展示），即便正文谎称
                # 「已删除」，用户也能在同一条消息上看到「被拒绝、未执行」的事实。
                emit(Progress(scope="check",
                              text=f"危险命令已被你拒绝，未执行：{params.command}",
                              status="error", key=f"denied:{params.command}"))
                raise ToolError(
                    f"命令未执行：用户拒绝了该操作（{danger.reason}）：{params.command}\n"
                    "该命令一行都没有运行，系统状态没有任何改变。"
                    "你必须如实告诉用户「该命令未执行，因为你拒绝了」——"
                    "绝对不要声称它已执行，也不要描述任何执行结果、影响或后续状态。"
                    "若该步骤因此无法完成，就说明卡在这里、并给出替代做法。")
        box = await self._sandbox.for_language("shell")   # run_shell 在 shell 容器执行
        meta = {"image": box.image} if getattr(box, "image", None) else None   # 供前端标注用的镜像
        try:
            res = await box.exec(["sh", "-c", params.command], self._timeout)
        except ToolError:
            raise
        except Exception as e:   # 容器级异常也带上镜像 meta，保证成功失败都显示镜像
            raise ToolError(str(e), meta=meta)
        out = format_exec(res, self._max_chars)
        if res.exit_code != 0 or res.timed_out:   # 非零退出/超时 → 标记失败（失败也带镜像 meta）
            raise ToolError(out, meta=meta)
        return ToolOutput(text=out, meta=meta) if meta else out
