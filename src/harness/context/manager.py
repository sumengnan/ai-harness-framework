from __future__ import annotations

from ..state import RunState
from ..types import Message, Role


class ContextManager:
    """决定每轮发给模型的消息列表。

    v1 极简：system_prompt + 完整历史。build() 是纯函数，
    未来的裁剪/压缩/RAG 注入都在这里加，loop 无感知。
    """

    def __init__(self, system_prompt: str) -> None:
        self._system_prompt = system_prompt

    def build(self, state: RunState) -> list[Message]:
        return [Message(role=Role.SYSTEM, content=self._system_prompt), *state.messages]
