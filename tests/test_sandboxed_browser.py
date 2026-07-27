import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import harness.net.policy as policy_mod
from harness.browser.base import PageResult
from harness.browser.factory import build_browser
from harness.browser.sandboxed_browser import SandboxedBrowser
from harness.config import HarnessConfig
from harness.tools.base import ToolExecutor, ToolRegistry
from harness.tools.builtins.browse_tool import BrowseTool
from harness.types import ToolCall


class FakeSandbox:
    """记录 write_file/exec 调用，read_file 返回预置输出（模拟容器内 runner 的产物）。"""

    workspace = "/workspace"

    def __init__(self, output, exit_code=0, stderr=""):
        self._output = output          # str：_browse_output.json 的内容；None=文件缺失
        self._exit_code = exit_code
        self._stderr = stderr
        self.writes: dict[str, str] = {}
        self.exec_calls: list[tuple[list[str], float]] = []

    async def start(self):
        pass

    async def close(self):
        pass

    async def write_file(self, path, content):
        self.writes[path] = content

    async def exec(self, command, timeout):
        self.exec_calls.append((command, timeout))
        return SimpleNamespace(stdout="", stderr=self._stderr,
                               exit_code=self._exit_code, timed_out=False)

    async def read_file(self, path):
        if self._output is None:
            raise RuntimeError(f"no such file: {path}")
        return self._output


def _ok_output(final_url="https://example.com/a", title="标题T",
               html="<html><body>hi</body></html>"):
    return json.dumps({"ok": True, "final_url": final_url, "title": title, "html": html})


async def test_fetch_ships_runner_and_policy_verbatim():
    sb = FakeSandbox(_ok_output())
    br = SandboxedBrowser(sb, allowed_domains=[], block_private=True,
                          launch_args=["--no-sandbox"])
    await br.fetch("https://example.com/a", timeout=5, wait_until="load")

    runner_src = (Path(policy_mod.__file__).parent.parent / "browser"
                  / "browse_runner.py").read_text()
    assert sb.writes["_browse_runner.py"] == runner_src
    # 关键：写进容器的 _policy.py 必须与宿主真实 policy.py 逐字节一致（防漂移）
    assert sb.writes["_policy.py"] == Path(policy_mod.__file__).read_text()


class _CountingSandbox(FakeSandbox):
    def __init__(self, output):
        super().__init__(output)
        self.started = 0
        self.closed = 0

    async def start(self):
        self.started += 1

    async def close(self):
        self.closed += 1


async def test_fetch_uses_ephemeral_browser_sub_when_factory_given():
    # 配了浏览器子沙箱工厂 → 抓取在一次性子沙箱内进行，基础容器不参与
    base = FakeSandbox(None)
    sub = _CountingSandbox(_ok_output(title="子沙箱标题"))
    br = SandboxedBrowser(base, allowed_domains=[], block_private=True,
                          sub_factory=lambda: sub)
    page = await br.fetch("https://example.com/a", timeout=5, wait_until="load")
    assert page.title == "子沙箱标题"
    assert sub.started == 1 and sub.closed == 1          # 一次性：起→用→销毁
    assert "_browse_runner.py" in sub.writes             # runner 写进了子沙箱
    assert base.writes == {} and base.exec_calls == []   # 基础容器未被用于抓取


async def test_fetch_playwright_missing_gives_actionable_hint():
    sb = FakeSandbox(json.dumps(
        {"ok": False, "error": "ModuleNotFoundError: No module named 'playwright'"}))
    br = SandboxedBrowser(sb, allowed_domains=[], block_private=True)
    with pytest.raises(RuntimeError) as ei:
        await br.fetch("https://example.com/a", timeout=5, wait_until="load")
    assert "HARNESS_BROWSER_SANDBOX_IMAGE" in str(ei.value)


async def test_fetch_writes_input_json_with_policy_params():
    sb = FakeSandbox(_ok_output())
    br = SandboxedBrowser(sb, allowed_domains=["example.com"], block_private=True,
                          user_agent="UA/1", launch_args=["--no-sandbox", "--x"])
    await br.fetch("https://example.com/a", timeout=7, wait_until="networkidle")

    params = json.loads(sb.writes["_browse_input.json"])
    assert params["url"] == "https://example.com/a"
    assert params["allowed_domains"] == ["example.com"]
    assert params["block_private"] is True
    assert params["wait_until"] == "networkidle"
    assert params["user_agent"] == "UA/1"
    assert params["launch_args"] == ["--no-sandbox", "--x"]


async def test_fetch_exec_command_and_timeout_margin():
    sb = FakeSandbox(_ok_output())
    br = SandboxedBrowser(sb, allowed_domains=[], block_private=True,
                          timeout_margin=10.0)
    await br.fetch("https://example.com/a", timeout=5, wait_until="load")

    cmd, timeout = sb.exec_calls[0]
    assert cmd == ["python3", "_browse_runner.py"]
    assert timeout == 15.0    # nav timeout + margin


async def test_fetch_parses_success_into_pageresult():
    sb = FakeSandbox(_ok_output(final_url="https://example.com/final",
                                title="落地页", html="<p>正文</p>"))
    br = SandboxedBrowser(sb, allowed_domains=[], block_private=True)
    page = await br.fetch("https://example.com/a", timeout=5, wait_until="load")
    assert isinstance(page, PageResult)
    assert page.final_url == "https://example.com/final"
    assert page.title == "落地页"
    assert page.html == "<p>正文</p>"


async def test_fetch_policy_error_raises():
    # 容器内逐跳校验拦截 → runner 写 ok=false 并非零退出
    sb = FakeSandbox(json.dumps({"ok": False, "error": "PolicyError: 内网地址"}),
                     exit_code=1)
    br = SandboxedBrowser(sb, allowed_domains=[], block_private=True)
    with pytest.raises(RuntimeError, match="PolicyError"):
        await br.fetch("http://internal/", timeout=5, wait_until="load")


async def test_fetch_missing_output_raises_with_stderr():
    # runner 崩溃、没写产物 → 用 stderr 兜底报错
    sb = FakeSandbox(None, exit_code=1, stderr="chromium 启动失败")
    br = SandboxedBrowser(sb, allowed_domains=[], block_private=True)
    with pytest.raises(RuntimeError, match="chromium 启动失败"):
        await br.fetch("https://example.com/a", timeout=5, wait_until="load")


async def test_url_validator_is_ignored_on_sandbox_path():
    # 宿主回调无法跨容器逐跳调用；策略在容器内执行，故传入必抛的 validator 不影响结果
    sb = FakeSandbox(_ok_output())
    br = SandboxedBrowser(sb, allowed_domains=[], block_private=True)

    def _always_raise(_u):
        raise AssertionError("host validator must not be called on sandbox path")

    page = await br.fetch("https://example.com/a", timeout=5, wait_until="load",
                          url_validator=_always_raise)
    assert page.title == "标题T"


async def test_browse_tool_surfaces_fetch_error_as_is_error():
    sb = FakeSandbox(json.dumps({"ok": False, "error": "PolicyError: 内网地址"}),
                     exit_code=1)
    br = SandboxedBrowser(sb, allowed_domains=[], block_private=True)
    tool = BrowseTool(br, allowed_domains=[], block_private=True, timeout=5,
                      wait_until="load", max_chars=8000,
                      resolve=lambda h: ["93.184.216.34"])   # 宿主入口预检放行
    reg = ToolRegistry(); reg.register(tool)
    ex = ToolExecutor(reg)
    r = await ex.execute(ToolCall(id="c1", name="browse",
                                  arguments={"url": "https://example.com/a"}))
    assert r.is_error is True


def test_factory_returns_sandboxed_when_sandbox_present():
    cfg = HarnessConfig(api_key="k", _env_file=None)
    sb = FakeSandbox(_ok_output())
    assert isinstance(build_browser(cfg, sb), SandboxedBrowser)


def test_factory_requires_sandbox():
    """浏览器统一走沙箱：无沙箱时 build_browser 明确报错，不再回退宿主本地 Playwright
    （宿主不再装 playwright，浏览器只在沙箱容器内跑）。"""
    import pytest
    cfg = HarnessConfig(api_key="k", _env_file=None)
    with pytest.raises(RuntimeError, match="沙箱"):
        build_browser(cfg)


async def test_sub_acquire_cached_box_not_closed_after_fetch():
    """sub_acquire 返回 cached=True（全局共用浏览器）→ 抓取后不销毁容器，生命周期归属主管理。"""
    box = _CountingSandbox(_ok_output())
    async def acquire():
        return box, True
    br = SandboxedBrowser(FakeSandbox(_ok_output()), allowed_domains=[], sub_acquire=acquire)
    await br.fetch("https://example.com/a", timeout=5, wait_until="load")
    assert box.closed == 0


async def test_sub_acquire_uncached_box_closed_after_fetch():
    """cached=False（未开缓存）→ 抓取后销毁（退回一次性语义）。"""
    box = _CountingSandbox(_ok_output())
    async def acquire():
        return box, False
    br = SandboxedBrowser(FakeSandbox(_ok_output()), allowed_domains=[], sub_acquire=acquire)
    await br.fetch("https://example.com/a", timeout=5, wait_until="load")
    assert box.closed == 1
