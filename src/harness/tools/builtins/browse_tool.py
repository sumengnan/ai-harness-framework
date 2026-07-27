# src/harness/tools/builtins/browse_tool.py
from __future__ import annotations

from urllib.parse import urlparse

from pydantic import BaseModel

from ..base import Tool
from ...browser.base import Browser
from ...browser.extract import extract_main_text
from ...net.policy import check_url
from ...net.sandbox_dns import resolve_in_sandbox
from ._sandbox_util import truncate


class BrowseTool(Tool):
    name = "browse"
    description = "用无头浏览器打开网页（含 JS 动态渲染）并提取正文，适合 http_request 抓不到内容的页面。"

    class Params(BaseModel):
        url: str

    def __init__(self, browser: Browser, allowed_domains, block_private: bool = True,
                 timeout: float = 30.0, wait_until: str = "networkidle",
                 max_chars: int = 8000, resolve=None, sandbox=None) -> None:
        self._browser = browser
        self._allowed = allowed_domains
        self._block_private = block_private
        self._timeout = timeout
        self._wait_until = wait_until
        self._max_chars = max_chars
        self._resolve_kw = {"resolve": resolve} if resolve is not None else {}
        # 有沙箱时浏览器在容器内出网，DNS 也须在容器内解析（与 http_request 对称，
        # 避免宿主/沙箱两侧 DNS 视图不一致导致 SSRF 绕过）
        self._sandbox = sandbox

    async def _check(self, url: str) -> None:
        if self._sandbox is not None and self._block_private:
            host = urlparse(url).hostname
            if host:
                ips = await resolve_in_sandbox(self._sandbox, host, self._timeout)
                check_url(url, self._allowed, self._block_private, resolve=lambda _h: ips)
                return
        check_url(url, self._allowed, self._block_private, **self._resolve_kw)

    async def run(self, params: "BrowseTool.Params") -> str:
        await self._check(params.url)   # PolicyError→is_error；有沙箱则用容器内 DNS

        def _validator(u: str) -> None:
            # 宿主浏览器逐跳校验用宿主 DNS；沙箱浏览器的逐跳校验由容器内 runner 依 _policy.py
            # 执行（此回调被 SandboxedBrowser 有意忽略），故这里保持宿主 resolve 语义即可。
            check_url(u, self._allowed, self._block_private, **self._resolve_kw)

        page = await self._browser.fetch(params.url, self._timeout, self._wait_until,
                                         url_validator=_validator)
        text = extract_main_text(page.html)
        if not text.strip():
            return f"（页面无可提取正文）标题：{page.title}"
        return truncate(f"标题：{page.title}\n最终URL：{page.final_url}\n\n{text}", self._max_chars)
