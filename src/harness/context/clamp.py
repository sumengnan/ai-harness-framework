# src/harness/context/clamp.py
from __future__ import annotations

from dataclasses import replace

from ..state import RunState
from ..types import Message
from ..usage import count_message_tokens

_OMIT = "\n…（此处内容因超出上下文预算已省略）…\n"


def _truncate_middle(text: str, keep_chars: int) -> str:
    """保头保尾、砍中段：judge 输入的末尾常是「只输出 JSON {...}」的格式指令，中段截断可保住它。"""
    if keep_chars <= 0:
        return _OMIT
    if len(text) <= keep_chars:
        return text
    head = keep_chars // 2
    return text[:head] + _OMIT + text[len(text) - (keep_chars - head):]


class ClampedContextManager:
    """包一层现有 ContextManager，在 build() 后按 (model, max_prompt_tokens) 确定性硬裁总输入：

    1) 先从最旧的中间消息开始丢（保 system 首条与本轮末条），直到不超上限；
    2) 仍超（如单轮 judge：system + 一条巨型 user）→ 对末条**文本**内容做中段截断（保头保尾）。

    仅确定性丢弃/截断，不摘要不检索。用于：
    - 编排器简单直答：base_ctx 按主模型预算裁过，再按**快速模型**口径重裁（快速模型窗口更小时防溢出）；
    - judge：给单轮 judge 的巨型输入（大段 grounding）加硬上限（防撑爆窗口/成本，是 fail-open 前的兜底）。
    max_prompt_tokens<=0 时透传不裁（默认关闭，零开销、行为不变）。"""

    def __init__(self, inner, model: str, max_prompt_tokens: int) -> None:
        self._inner = inner
        self._model = model
        self._cap = int(max_prompt_tokens or 0)

    def build(self, state: RunState) -> list[Message]:
        msgs = list(self._inner.build(state))
        if self._cap <= 0 or not msgs:
            return msgs
        # 1) 丢最旧的中间消息（index 1 起），保 system(首) 与末条本轮消息
        while len(msgs) > 2 and count_message_tokens(msgs, self._model) > self._cap:
            del msgs[1]
        # 2) 仍超 → 中段截断末条**文本**内容（多模态 list 内容不动，避免破坏结构）
        last = msgs[-1]
        if isinstance(last.content, str) \
                and count_message_tokens(msgs, self._model) > self._cap:
            others = count_message_tokens(msgs[:-1], self._model) if len(msgs) > 1 else 0
            budget = max(self._cap - others, 0)
            tok = count_message_tokens([last], self._model)
            if tok > budget and tok > 0:
                cpt = max(len(last.content) / tok, 1.0)    # 每 token 约几字符
                keep = int(budget * cpt * 0.85)            # 先按估算截，留 15% 余量
                new = replace(last, content=_truncate_middle(last.content, keep))
                # 估算偏大时再收几次，确保确定性不超上限（罕见路径，多收几次无妨）
                for _ in range(5):
                    if keep <= 0 or count_message_tokens(msgs[:-1] + [new], self._model) <= self._cap:
                        break
                    keep = int(keep * 0.7)
                    new = replace(last, content=_truncate_middle(last.content, keep))
                msgs[-1] = new
        return msgs
