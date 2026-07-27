import os

import pytest

from harness.browser.sandboxed_browser import SandboxedBrowser
from harness.config import HarnessConfig
from harness.net.policy import PolicyError
from harness.sandbox.factory import build_sandbox

# 需真实远程 docker + 含 Playwright/Chromium 的镜像才跑；CI 无这些环境时自动跳过。
_HAS_ENV = bool(os.getenv("HARNESS_SANDBOX_DOCKER_HOST") and os.getenv("HARNESS_SANDBOX_IMAGE"))
pytestmark = pytest.mark.skipif(
    not _HAS_ENV,
    reason="需要真实远程 docker + 浏览器镜像（HARNESS_SANDBOX_DOCKER_HOST + HARNESS_SANDBOX_IMAGE）")


def _sandbox():
    cfg = HarnessConfig(
        api_key="k", sandbox_backend="docker",
        sandbox_docker_host=os.environ["HARNESS_SANDBOX_DOCKER_HOST"],
        sandbox_network=os.getenv("HARNESS_SANDBOX_NETWORK", "bridge"),
        sandbox_read_only=False,
        sandbox_docker_tls_ca_cert=os.environ.get("HARNESS_SANDBOX_DOCKER_TLS_CA_CERT", ""),
        sandbox_docker_tls_client_cert=os.environ.get("HARNESS_SANDBOX_DOCKER_TLS_CLIENT_CERT", ""),
        sandbox_docker_tls_client_key=os.environ.get("HARNESS_SANDBOX_DOCKER_TLS_CLIENT_KEY", ""))
    # 浏览器测试直接用一个含 Playwright/Chromium 的镜像起单容器沙箱
    return build_sandbox(cfg, image=os.environ["HARNESS_SANDBOX_IMAGE"])


async def test_sandboxed_browser_fetches_real_page():
    sb = _sandbox()
    await sb.start()
    try:
        br = SandboxedBrowser(sb, allowed_domains=[], block_private=True)
        page = await br.fetch("https://example.com/", timeout=30, wait_until="load")
        assert page.title
        assert page.html
    finally:
        await sb.close()


async def test_sandboxed_browser_blocks_internal_url():
    sb = _sandbox()
    await sb.start()
    try:
        br = SandboxedBrowser(sb, allowed_domains=[], block_private=True)
        with pytest.raises((RuntimeError, PolicyError)):
            await br.fetch("http://169.254.169.254/", timeout=30, wait_until="load")
    finally:
        await sb.close()
