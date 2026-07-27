from __future__ import annotations

from dataclasses import dataclass

from ..types import Message, Role
from ..usage import count_message_tokens


@dataclass
class WindowResult:
    """滑动窗口裁剪结果。kept 为保留的最近若干整轮，evicted 为被挤出的更早消息。"""
    kept: list[Message]
    evicted: list[Message]


def _strip_leading_orphans(msgs: list[Message]) -> tuple[list[Message], list[Message]]:
    """防御性清理：窗口首部若出现孤儿 tool 结果（无前置 assistant tool_call），剥掉。

    正常情况下窗口从 user 边界开始不会有孤儿；此函数仅兜底异常历史。
    """
    i = 0
    while i < len(msgs) and msgs[i].role == Role.TOOL:
        i += 1
    return msgs[i:], msgs[:i]


class WindowStrategy:
    """L1 滑动窗口：按 token 预算保留最近的完整轮次，切割只落在干净的 user 边界。

    turn 边界定义为 role==user（且非 tool 续接）的消息。绝不从一轮中间切开——
    否则会留下孤儿 tool 结果或 tool_calls 被截断的 assistant，OpenAI 会 400。
    """

    def __init__(self, model: str) -> None:
        self._model = model

    def select(self, history: list[Message], budget_tokens: int) -> WindowResult:
        if not history:
            return WindowResult(kept=[], evicted=[])
        starts = [i for i, m in enumerate(history)
                  if m.role == Role.USER and m.tool_call_id is None]
        if not starts:
            # 无 user 边界，无法安全切割 → 全保留（罕见）。
            return WindowResult(kept=list(history), evicted=[])
        # 从最新边界往前扩，取能放下的最大后缀；至少保留最后一个完整轮。
        chosen = starts[-1]
        for start in reversed(starts):
            if count_message_tokens(history[start:], self._model) <= budget_tokens:
                chosen = start
            else:
                break
        kept, orphan_evicted = _strip_leading_orphans(list(history[chosen:]))
        evicted = list(history[:chosen]) + orphan_evicted
        return WindowResult(kept=kept, evicted=evicted)
