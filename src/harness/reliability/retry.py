# src/harness/reliability/retry.py
from __future__ import annotations

import asyncio
import random
from typing import AsyncIterator, Awaitable, Callable

from opentelemetry import trace

from ..llm.base import ModelClient, StreamChunk
from ..types import Message

try:  # 真实运行时用 openai 的瞬时错误类型
    from openai import (
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
    )

    _DEFAULT_TRANSIENT: tuple[type[BaseException], ...] = (
        APITimeoutError,
        APIConnectionError,
        RateLimitError,
        InternalServerError,
    )
except Exception:  # openai 未安装时的降级（正常环境不会触发）
    _DEFAULT_TRANSIENT = ()


class RetryingModelClient:
    """装饰任意 ModelClient，对瞬时错误做指数退避重试。

    安全约束：只在流尚未产出任何 chunk 前失败才重试；中途断裂直接抛出，
    避免重复 yield 半截输出。仍实现 ModelClient 协议，loop 无感知。
    """

    def __init__(
        self,
        inner: ModelClient,
        max_retries: int = 2,
        base_delay: float = 0.5,
        transient: tuple[type[BaseException], ...] = _DEFAULT_TRANSIENT,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._inner = inner
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._transient = transient
        self._sleep = sleep

    async def stream(
        self, messages: list[Message], tools: list[dict]
    ) -> AsyncIterator[StreamChunk]:
        for attempt in range(1, self._max_retries + 2):  # 1 次正常 + max_retries 次重试
            produced = False
            try:
                async for chunk in self._inner.stream(messages, tools):
                    produced = True
                    if chunk.type == "done":
                        chunk.attempts = attempt
                    yield chunk
                return
            except self._transient as e:
                if produced or attempt > self._max_retries:
                    raise
                trace.get_current_span().add_event(
                    "model_call.retry",
                    {"attempt": attempt, "error": type(e).__name__},
                )
                delay = self._base_delay * 2 ** (attempt - 1)
                delay += random.uniform(0, self._base_delay * 0.1)  # 抖动
                await self._sleep(delay)
