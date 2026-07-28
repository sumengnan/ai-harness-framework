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
        self._base_wall = 0.0     # 断点续跑时带回来的、之前已消耗的墙钟

    def start(self) -> None:
        if self._start is None:   # 幂等：只在首次设墙钟基准（多 agent 共享 budget）
            self._start = self._clock()

    def restore(self, total_tokens: int = 0, wall_seconds: float = 0.0) -> None:
        """把之前那次 run 已消耗的量装回来，供 resume 接着算。

        必须在 start() 之前调用（start() 之后调用只影响后续累计，墙钟基准已定）。
        崩溃到重启之间的停机时间**不计入**——只累计真正在跑的时间。

        只用于顶层 resume：共享 budget 的子 agent 不该调用它，否则会覆盖兄弟
        节点已累计的量。
        """
        self._total_tokens = total_tokens
        self._base_wall = wall_seconds

    def add_usage(self, usage: Usage) -> None:
        self._total_tokens += usage.total_tokens

    def check(self) -> None:
        if self._max_tokens is not None and self._total_tokens > self._max_tokens:
            raise BudgetExceeded(f"token 预算超限：{self._total_tokens} > {self._max_tokens}")
        if self._max_wall is not None and self._start is not None:
            elapsed = self.elapsed_seconds
            if elapsed > self._max_wall:
                raise BudgetExceeded(f"时间预算超限：{elapsed:.1f}s > {self._max_wall}s")

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def elapsed_seconds(self) -> float:
        """已消耗墙钟：本次 run 的 + restore() 带回来的。未 start() 则只有后者。"""
        if self._start is None:
            return self._base_wall
        return self._base_wall + (self._clock() - self._start)
