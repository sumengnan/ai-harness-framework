# src/harness/browser/sandboxed_browser.py
from __future__ import annotations

import json
from pathlib import Path

from .base import PageResult
from ..net import policy as _policy_mod

# Chromium 在 cap_drop=ALL + no-new-privileges + 非 root + 小 /dev/shm 下的必备参数
DEFAULT_LAUNCH_ARGS = [
    "--no-sandbox", "--disable-setuid-sandbox",
    "--disable-dev-shm-usage", "--disable-gpu",
]

_RUNNER_SRC = Path(__file__).with_name("browse_runner.py").read_text()
_POLICY_SRC = Path(_policy_mod.__file__).read_text()   # 逐字节副本 → 容器内 _policy.py，零漂移


class SandboxedBrowser:
    """在沙箱容器内跑无头 Chromium 抓网页（实现 Browser 协议）。

    与 SandboxedHttpRequestTool 对称：渲染/出网/DNS 解析都发生在容器内；SSRF 逐跳
    校验用宿主 net/policy.py 的逐字节副本在容器内执行。不拥有容器——容器由 assembly 属主创建/销毁。

    安全：fetch 的 url_validator 参数仅为满足 Browser 协议而保留，被有意忽略——宿主回调无法
    跨容器逐跳调用；等价策略由容器内 runner 依据 allowed_domains/block_private 执行。
    """

    def __init__(self, sandbox, allowed_domains, block_private: bool = True,
                 user_agent: str = "", launch_args=None,
                 timeout_margin: float = 10.0, sub_factory=None, sub_acquire=None) -> None:
        self._sandbox = sandbox
        self._allowed = list(allowed_domains)
        self._block_private = block_private
        self._user_agent = user_agent
        self._launch_args = list(launch_args) if launch_args else list(DEFAULT_LAUNCH_ARGS)
        self._timeout_margin = timeout_margin
        # sub_acquire：async ()->(box, cached) 的缓存提供者（优先）。cached=True 表示该浏览器子沙箱
        # 由属主（SandboxManager）按会话缓存复用、空闲回收——本类**不得**销毁它。
        # sub_factory：无缓存时的旧式一次性子沙箱工厂（用完即销毁）。二者皆无则复用基础容器。
        self._sub_acquire = sub_acquire
        self._sub_factory = sub_factory
        self._provisioned = False

    async def start(self) -> None:
        await self._sandbox.start()   # 幂等；容器可能已由其他工具启动

    async def close(self) -> None:
        pass   # 共享容器由其属主（assembly/sandbox）关闭

    async def _provision_into(self, box) -> None:
        await box.write_file("_browse_runner.py", _RUNNER_SRC)
        await box.write_file("_policy.py", _POLICY_SRC)

    async def _fetch_in(self, box, url: str, timeout: float, wait_until: str) -> PageResult:
        await box.write_file("_browse_input.json", json.dumps({
            "url": url, "timeout": timeout, "wait_until": wait_until,
            "allowed_domains": self._allowed, "block_private": self._block_private,
            "user_agent": self._user_agent, "launch_args": self._launch_args,
        }))
        res = await box.exec(["python3", "_browse_runner.py"], timeout + self._timeout_margin)
        out = None
        try:
            out = json.loads(await box.read_file("_browse_output.json"))
        except Exception:
            out = None   # runner 崩溃前未落盘 → 下面用 stderr 兜底
        if not out or not out.get("ok"):
            err = out.get("error") if out else (res.stderr.strip() or f"browse 失败（exit {res.exit_code}）")
            if "playwright" in err.lower():
                err += ("；浏览器沙箱镜像缺少 Playwright——请设 HARNESS_BROWSER_SANDBOX_IMAGE "
                        "为含 Playwright+Chromium+curl 的镜像，或让基础镜像自带 playwright")
            raise RuntimeError(err)
        return PageResult(final_url=out["final_url"], title=out["title"], html=out["html"])

    async def fetch(self, url: str, timeout: float, wait_until: str,
                    url_validator=None) -> PageResult:
        # 优先用缓存提供者：同会话浏览器子沙箱按 1h 空闲复用，不随手销毁（避免每次重建 Chromium 容器）
        if self._sub_acquire is not None:
            box, cached = await self._sub_acquire()
            await box.start()                  # 幂等；复用时不会重复启动
            try:
                await self._provision_into(box)   # 幂等地写入 runner（浏览器进程仍每次抓取新起）
                return await self._fetch_in(box, url, timeout, wait_until)
            finally:
                if not cached:                 # 缓存关闭时才销毁；缓存复用的归属主管理
                    try:
                        await box.close()
                    except Exception:
                        pass
        if self._sub_factory is not None:
            box = self._sub_factory()          # 一次性浏览器子沙箱（专用 playwright 镜像）
            await box.start()
            try:
                await self._provision_into(box)
                return await self._fetch_in(box, url, timeout, wait_until)
            finally:
                try:
                    await box.close()          # 用完即销毁
                except Exception:
                    # 容器可能已 OOM 消失，移除报 404/409 不应掩盖真正的抓取错误
                    pass
        # 复用会话基础容器（需基础镜像自带 playwright）
        if not self._provisioned:
            await self._provision_into(self._sandbox)
            self._provisioned = True
        return await self._fetch_in(self._sandbox, url, timeout, wait_until)
