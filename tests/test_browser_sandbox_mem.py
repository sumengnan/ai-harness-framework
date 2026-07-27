from harness.config import HarnessConfig
from harness.sandbox.factory import _docker_for


def _cfg():
    return HarnessConfig(sandbox_docker_host="tcp://x:2376",
                         sandbox_mem_limit="100m", browser_sandbox_mem_limit="1g")


def test_docker_for_defaults_to_base_mem_limit():
    box = _docker_for(_cfg(), "img")
    assert box._mem_limit == "100m"


def test_docker_for_mem_limit_override():
    # 浏览器子沙箱用更大额度，避免 Chromium 被 OOM 杀掉
    cfg = _cfg()
    box = _docker_for(cfg, cfg.browser_sandbox_image or "pw-img",
                      mem_limit=cfg.browser_sandbox_mem_limit)
    assert box._mem_limit == "1g"


def test_browser_sandbox_mem_limit_default():
    assert HarnessConfig().browser_sandbox_mem_limit == "1g"
