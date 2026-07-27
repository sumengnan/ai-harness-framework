from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Protocol, runtime_checkable

from ..types import Message
from ..usage import Usage


@dataclass
class ToolCallDelta:
    index: int
    id: str | None = None
    name: str | None = None
    arguments: str | None = None  # 部分 JSON 字符串片段，跨 chunk 累加


@dataclass
class StreamChunk:
    type: str  # "text" | "reasoning" | "tool_call" | "done"
    text: str | None = None
    tool_call_delta: ToolCallDelta | None = None
    usage: Usage | None = None      # 仅 done chunk 携带
    attempts: int = 1               # 仅 done chunk 携带（重试次数）


@runtime_checkable
class ModelClient(Protocol):
    async def stream(
        self, messages: list[Message], tools: list[dict]
    ) -> AsyncIterator[StreamChunk]:
        ...
