# src/harness/reliability/budget.py
from __future__ import annotations

import time
from typing import Callable

from ..usage import Usage


class BudgetExceeded(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class BudgetTracker:
    """纯累计器：累计 token 与墙钟时间，check() 超限即抛 BudgetExceeded。

    clock 可注入以便测试（默认 time.monotonic）。

    注意：未调用 start() 时，max_wall_seconds 检查会被静默跳过（无起始时刻可比较）；
    请在使用前先调用 start()。

    start() 幂等，全树只在根 run 设一次墙钟基准；每个顶层 run 应用新的
    BudgetTracker 实例。
    """

    def __init__(
        self,
        max_tokens: int | None = None,
        max_wall_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_tokens = max_tokens
        self._max_wall = max_wall_seconds
        self._clock = clock
        self._start: float | None = None
        self._total_tokens = 0

    def start(self) -> None:
        if self._start is None:   # 幂等：只在首次设墙钟基准（多 agent 共享 budget）
            self._start = self._clock()

    def add_usage(self, usage: Usage) -> None:
        self._total_tokens += usage.total_tokens

    def check(self) -> None:
        if self._max_tokens is not None and self._total_tokens > self._max_tokens:
            raise BudgetExceeded(f"token 预算超限：{self._total_tokens} > {self._max_tokens}")
        if self._max_wall is not None and self._start is not None:
            elapsed = self._clock() - self._start
            if elapsed > self._max_wall:
                raise BudgetExceeded(f"时间预算超限：{elapsed:.1f}s > {self._max_wall}s")

    @property
    def total_tokens(self) -> int:
        return self._total_tokens
