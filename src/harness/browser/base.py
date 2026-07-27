# src/harness/browser/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable


@dataclass
class PageResult:
    final_url: str
    title: str
    html: str


@runtime_checkable
class Browser(Protocol):
    async def start(self) -> None: ...
    async def close(self) -> None: ...
    async def fetch(self, url: str, timeout: float, wait_until: str,
                    url_validator: Callable[[str], None] | None = None) -> PageResult: ...
