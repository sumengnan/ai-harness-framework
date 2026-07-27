# src/harness/browser/fake.py
from __future__ import annotations

from .base import PageResult


class FakeBrowser:
    """测试用 Browser：返回预设 HTML，不下载 chromium、不联网。"""

    def __init__(self, pages: dict[str, tuple[str, str]],
                 redirects: dict[str, str] | None = None) -> None:
        self._pages = pages          # {url: (title, html)}
        self._redirects = redirects or {}   # {url: 下一跳 url}
        self.started = False

    async def start(self) -> None:
        self.started = True

    async def close(self) -> None:
        self.started = False

    async def fetch(self, url: str, timeout: float, wait_until: str,
                    url_validator=None) -> PageResult:
        # 沿重定向链逐跳走，每跳（含初始与落地）都校验，抛则冒泡（模拟浏览器逐跳 SSRF 拦截）。
        seen: set[str] = set()
        current = url
        while True:
            if url_validator is not None:
                url_validator(current)
            if current in self._redirects and current not in seen:
                seen.add(current)
                current = self._redirects[current]
                continue
            break
        if current not in self._pages:
            raise RuntimeError(f"FakeBrowser 无预设页面：{current}")
        title, html = self._pages[current]
        return PageResult(final_url=current, title=title, html=html)
