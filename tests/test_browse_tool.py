from types import SimpleNamespace

import pytest

from harness.browser.fake import FakeBrowser
from harness.tools.base import ToolRegistry, ToolExecutor
from harness.tools.builtins.browse_tool import BrowseTool
from harness.types import ToolCall


class _FakeSandbox:
    """模拟沙箱：exec(python3 -c 解析脚本 host) 返回容器内 DNS 结果。"""

    def __init__(self, ip_map):
        self._ip_map = ip_map
        self.hosts = []

    async def exec(self, cmd, timeout, *, quiet=False):
        host = cmd[-1]                       # ["python3","-c",script,host]
        self.hosts.append(host)
        return SimpleNamespace(
            stdout="\n".join(self._ip_map.get(host, [])), stderr="", exit_code=0)

ARTICLE = (
    "<html><head><title>标题T</title></head><body>"
    "<header><nav>首页 关于 联系我们 登录 注册</nav></header>"
    "<article><p>这是一段足够长的正文内容，用于验证浏览器工具能够正确渲染并提取页面主体文字，"
    "而不是把导航栏和页脚一起塞进结果里，从而保证喂给知识库的内容是干净的。这段内容需要再长一些，"
    "以便让提取算法能够明显区分正文段落与周围的导航、页脚等样板内容，从而提升提取的稳定性和准确性。</p></article>"
    "<footer>版权所有 2026 隐私政策 网站地图</footer></body></html>"
)


def _tool(pages, resolve):
    fb = FakeBrowser(pages)
    return BrowseTool(fb, allowed_domains=[], block_private=True, timeout=5,
                      wait_until="load", max_chars=8000, resolve=resolve)


async def test_browse_returns_extracted_text():
    tool = _tool({"http://example.com/a": ("标题T", ARTICLE)},
                 resolve=lambda h: ["93.184.216.34"])
    out = await tool.run(tool.Params(url="http://example.com/a"))
    assert "标题T" in out
    assert "足够长的正文内容" in out
    assert "登录 注册" not in out


async def test_browse_ssrf_is_error():
    tool = _tool({}, resolve=lambda h: ["127.0.0.1"])
    reg = ToolRegistry(); reg.register(tool)
    ex = ToolExecutor(reg)
    r = await ex.execute(ToolCall(id="c1", name="browse",
                                  arguments={"url": "http://internal/"}))
    assert r.is_error is True


async def test_browse_metadata_ip_literal_blocked():
    tool = _tool({}, resolve=None)   # 用真实 default_resolve；IP 字面量不走网络
    reg = ToolRegistry(); reg.register(tool)
    ex = ToolExecutor(reg)
    r = await ex.execute(ToolCall(id="c1", name="browse",
                                  arguments={"url": "http://169.254.169.254/"}))
    assert r.is_error is True


async def test_browse_redirect_to_internal_blocked():
    # 初始 example.com 公网、重定向跳到元数据地址 → 逐跳校验应拦截
    fb = FakeBrowser({"http://example.com/a": ("落地", ARTICLE)},
                     redirects={"http://example.com/a": "http://169.254.169.254/"})
    tool = BrowseTool(fb, allowed_domains=[], block_private=True, timeout=5,
                      wait_until="load", max_chars=8000,
                      resolve=lambda h: ["93.184.216.34"] if h == "example.com" else [h])
    reg = ToolRegistry(); reg.register(tool)
    ex = ToolExecutor(reg)
    r = await ex.execute(ToolCall(id="c1", name="browse",
                                  arguments={"url": "http://example.com/a"}))
    assert r.is_error is True   # 重定向跳被策略拦截


async def test_browse_dns_resolved_in_sandbox():
    # 有沙箱时主机名 DNS 下沉到容器解析：确认走了 sandbox.exec 而非宿主 resolve，且放行
    # （FakeBrowser 的逐跳 url_validator 用宿主 resolve，真实 SandboxedBrowser 会忽略它、
    #  由容器内 runner 校验，故此处宿主 resolve 也置公网以免干扰）
    fb = FakeBrowser({"http://example.com/a": ("标题T", ARTICLE)})
    sb = _FakeSandbox({"example.com": ["93.184.216.34"]})
    tool = BrowseTool(fb, allowed_domains=[], block_private=True, timeout=5,
                      wait_until="load", max_chars=8000,
                      resolve=lambda h: ["93.184.216.34"], sandbox=sb)
    out = await tool.run(tool.Params(url="http://example.com/a"))
    assert "标题T" in out
    assert sb.hosts == ["example.com"]        # 确实走了沙箱内 DNS 解析


async def test_browse_sandbox_dns_internal_blocked():
    # 沙箱解析到内网 → 前置校验以沙箱 DNS 为准拦截（即便宿主看似公网），fetch 都不会发生
    fb = FakeBrowser({})
    sb = _FakeSandbox({"evil.com": ["10.0.0.5"]})
    tool = BrowseTool(fb, allowed_domains=[], block_private=True, timeout=5,
                      wait_until="load", max_chars=8000,
                      resolve=lambda h: ["93.184.216.34"], sandbox=sb)
    reg = ToolRegistry(); reg.register(tool)
    ex = ToolExecutor(reg)
    r = await ex.execute(ToolCall(id="c1", name="browse",
                                  arguments={"url": "http://evil.com/"}))
    assert r.is_error is True


async def test_browse_empty_content_message():
    tool = _tool({"http://example.com/e": ("空页", "<html><body></body></html>")},
                 resolve=lambda h: ["93.184.216.34"])
    out = await tool.run(tool.Params(url="http://example.com/e"))
    assert "无可提取正文" in out


LONG_ARTICLE = (
    "<html><head><title>长文</title></head><body>"
    "<header><nav>首页 关于 联系我们 登录 注册</nav></header>"
    "<article><p>"
    + "光合作用是绿色植物利用光能把二氧化碳和水转化为储存能量的有机物并释放氧气的过程" * 20
    + "</p></article>"
    "<footer>版权所有 2026 隐私政策 网站地图</footer></body></html>"
)


async def test_browse_truncates_long_text():
    fb = FakeBrowser({"http://example.com/long": ("长文", LONG_ARTICLE)})
    tool = BrowseTool(fb, allowed_domains=[], block_private=True, timeout=5,
                      wait_until="load", max_chars=50,
                      resolve=lambda h: ["93.184.216.34"])
    out = await tool.run(tool.Params(url="http://example.com/long"))
    assert out.endswith("…(已截断)")
