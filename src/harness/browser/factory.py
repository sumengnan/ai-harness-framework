# src/harness/browser/factory.py
from __future__ import annotations


def build_browser(config, sandbox=None, sub_factory=None, sub_acquire=None):
    """浏览器统一在**沙箱容器**内跑无头 Chromium（容器镜像自带 Playwright+Chromium）。
    宿主不再内置 Playwright，故无沙箱时明确报错，不回退宿主本地 Playwright。

    sub_acquire：async ()->(box, cached) 的缓存提供者（浏览器子沙箱按会话缓存复用，优先）。
    sub_factory：旧式一次性子沙箱工厂（用完即销毁）。
    """
    if sandbox is None:
        raise RuntimeError(
            "浏览器抓取需要沙箱：浏览器在沙箱容器内跑（须开 enable_sandbox 并配浏览器专用镜像），"
            "宿主已不再内置 Playwright。")
    from .sandboxed_browser import SandboxedBrowser
    return SandboxedBrowser(
        sandbox, config.http_allowed_domains, config.http_block_private,
        config.browser_user_agent, config.sandbox_browser_launch_args,
        sub_factory=sub_factory, sub_acquire=sub_acquire)
