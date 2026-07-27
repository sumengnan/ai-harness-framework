# src/harness/tools/builtins/sandbox_http_tool.py
from __future__ import annotations

from urllib.parse import urljoin, urlparse

from pydantic import BaseModel

from ..base import Tool
from ...net.policy import PolicyError, check_url
from ...net.sandbox_dns import resolve_in_sandbox
from .http_tool import (
    browser_fallback_or_none, looks_blocked, merge_user_agent, render_http_result)


def _parse_response(raw: str) -> tuple[int, str | None, str, str]:
    """解析 curl -i 的输出：返回 (status, location, content_type, body)。

    curl -i 会把 1xx 信息响应（100 Continue、以及 CDN 为预加载 CSS/JS 常发的 103 Early Hints）
    的头块也打印在最终响应之前。必须逐个跳过这些 1xx 头块、取第一个非 1xx 头块为最终响应，
    否则会把 103 当成状态码、content-type 解析成空——于是 HTML 页面识别不出、被原样打印。
    """
    rest = raw
    status = 0
    location: str | None = None
    content_type = ""
    while True:
        sep = "\r\n\r\n" if "\r\n\r\n" in rest else "\n\n"
        head, found, body = rest.partition(sep)
        lines = head.splitlines()
        if not (lines and lines[0].startswith("HTTP/")):
            return status, location, content_type, rest   # 已越过所有头块（不含状态行）
        status = 0
        parts = lines[0].split()
        if len(parts) >= 2 and parts[1].isdigit():
            status = int(parts[1])
        location, content_type = None, ""
        for ln in lines[1:]:
            low = ln.lower()
            if location is None and low.startswith("location:"):
                location = ln.split(":", 1)[1].strip()
            elif not content_type and low.startswith("content-type:"):
                content_type = ln.split(":", 1)[1].strip()
        if 100 <= status < 200 and found:      # 1xx 信息响应：跳过，继续解析下一头块
            rest = body
            continue
        return status, location, content_type, body


class SandboxedHttpRequestTool(Tool):
    name = "http_request"
    description = ("发起 HTTP(S) 请求抓取网页或调用外部 API（在沙箱容器内用 curl 执行，"
                   "出网来自沙箱）。默认可访问公网，禁止内网地址。"
                   "网页默认返回解析后的标题+正文；需要原始 HTML 时传 raw=true。"
                   "抓取失败或页面疑似被防抓/需 JS 渲染时，会自动改用无头浏览器重试（若已启用）。")

    class Params(BaseModel):
        url: str
        method: str = "GET"
        headers: dict | None = None
        body: str | None = None
        raw: bool = False

    def __init__(self, sandbox, allowed_domains, block_private: bool = True,
                 timeout: float = 30.0, max_bytes: int = 5_000_000,
                 max_redirects: int = 5, browser_fallback=None,
                 user_agent: str = "") -> None:
        self._sandbox = sandbox
        self._allowed = allowed_domains
        self._block_private = block_private
        self._timeout = timeout
        self._max_bytes = max_bytes
        self._max_redirects = max_redirects
        self._browser_fallback = browser_fallback
        self._user_agent = user_agent

    def set_browser_fallback(self, fn) -> None:
        self._browser_fallback = fn

    async def _resolve_in_sandbox(self, host: str) -> list[str]:
        return await resolve_in_sandbox(self._sandbox, host, self._timeout)

    async def run(self, params: "SandboxedHttpRequestTool.Params") -> str:
        try:
            status, final_url, ctype, body, suffix = await self._fetch(params)
        except PolicyError:
            raise                       # 安全拦截：不兜底
        except Exception as e:          # noqa: BLE001  出错 → 尝试浏览器兜底
            fb = await browser_fallback_or_none(
                self._browser_fallback, params.url, f"http_request 出错：{e}")
            if fb is not None:
                return fb
            raise
        if not params.raw and looks_blocked(status, body):
            fb = await browser_fallback_or_none(
                self._browser_fallback, final_url, f"HTTP {status} 疑似防抓或需 JS 渲染")
            if fb is not None:
                return fb
        return render_http_result(status, final_url, ctype, body, suffix, params.raw)

    async def _fetch(self, params: "SandboxedHttpRequestTool.Params") -> tuple[int, str, str, str, str]:
        url = params.url
        for _ in range(self._max_redirects + 1):
            parsed = urlparse(url)
            host = parsed.hostname
            pinned_ips: list[str] | None = None
            if self._block_private and host:
                # 执行在沙箱：DNS 在容器内解析；决策在宿主：把 IP 传回 check_url 校验
                ips = await self._resolve_in_sandbox(host)
                check_url(url, self._allowed, self._block_private, resolve=lambda _h: ips)
                pinned_ips = ips
            else:
                check_url(url, self._allowed, self._block_private)

            cmd = ["curl", "-s", "-S", "-i", "--max-time", str(max(1, int(self._timeout))),
                   "-X", params.method]
            if pinned_ips:
                # 把 curl 钉到宿主已校验过的 IP，避免容器再解析一次（关闭 rebinding 时间窗）
                port = parsed.port or (443 if parsed.scheme == "https" else 80)
                cmd += ["--resolve", f"{host}:{port}:{','.join(pinned_ips)}"]
            # 与宿主版同一套合并规则：显式传的 User-Agent 优先，否则补默认 UA
            for k, v in (merge_user_agent(params.headers, self._user_agent) or {}).items():
                cmd += ["-H", f"{k}: {v}"]
            if params.body is not None:
                cmd += ["--data-binary", params.body]
            cmd.append(url)

            res = await self._sandbox.exec(cmd, self._timeout + 5)
            if not res.stdout:
                raise RuntimeError(
                    f"curl 无输出（exit {res.exit_code}）："
                    f"{res.stderr.strip() or '容器内可能缺少 curl，请使用含 curl 的镜像'}")
            status, location, ctype, body = _parse_response(res.stdout)
            if 300 <= status < 400 and location:
                url = urljoin(url, location)
                continue
            suffix = "…(已截断)" if len(body) > self._max_bytes else ""
            body = body[: self._max_bytes]
            return status, url, ctype, body, suffix
        raise RuntimeError(f"超过最大重定向次数（{self._max_redirects}）")
