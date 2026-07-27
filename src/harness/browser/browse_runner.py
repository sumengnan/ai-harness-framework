"""在沙箱容器内执行的无头浏览器抓取脚本。

由 SandboxedBrowser 写入容器工作区后以 `python3 _browse_runner.py` 执行。

约束：本脚本运行在容器内，sys.path 与宿主不同——**只允许**依赖标准库、`playwright`
以及同目录被写进来的 `_policy`（宿主 net/policy.py 的逐字节副本）。绝不 import `harness` 包。

流程：读 _browse_input.json → 用 sync Playwright 启 Chromium → 对每一跳导航请求用
_policy.check_url 做 SSRF 校验（与宿主策略（SandboxedBrowser）一致，子资源不校验）→
把 {final_url,title,html} 写进 _browse_output.json；任何异常写 {"ok":false,"error":...} 并非零退出。
"""
from __future__ import annotations

import json
import sys

INPUT = "_browse_input.json"
OUTPUT = "_browse_output.json"
HTML_MAX_BYTES = 5_000_000   # 防御性上限，护住 64MB tmpfs 工作区


def _write_output(obj) -> None:
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def main() -> int:
    with open(INPUT, encoding="utf-8") as f:
        params = json.load(f)

    url = params["url"]
    timeout = float(params.get("timeout", 30.0))
    wait_until = params.get("wait_until", "networkidle")
    allowed_domains = params.get("allowed_domains", [])
    block_private = bool(params.get("block_private", True))
    user_agent = params.get("user_agent") or None
    launch_args = list(params.get("launch_args", []))

    from _policy import PolicyError, check_url  # noqa: E402  容器内同目录副本
    from playwright.sync_api import sync_playwright  # noqa: E402

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=launch_args)
        try:
            context = browser.new_context(accept_downloads=False, user_agent=user_agent)

            def _route(route):
                req = route.request
                if req.is_navigation_request():
                    try:
                        # 逐跳 SSRF：容器内出网，故容器内解析 DNS 才是正确视角
                        check_url(req.url, allowed_domains, block_private)
                    except PolicyError:
                        route.abort()
                        return
                route.continue_()

            context.route("**/*", _route)
            page = context.new_page()
            page.goto(url, wait_until=wait_until, timeout=timeout * 1000)
            html = page.content()
            if len(html.encode()) > HTML_MAX_BYTES:
                html = html.encode()[:HTML_MAX_BYTES].decode(errors="ignore")
            _write_output({"ok": True, "final_url": page.url,
                           "title": page.title(), "html": html})
        finally:
            browser.close()   # 长生命周期容器复用，务必收尾防进程泄漏
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:   # noqa: BLE001  任何失败都要落盘并非零退出
        _write_output({"ok": False, "error": f"{type(e).__name__}: {e}"})
        sys.exit(1)
