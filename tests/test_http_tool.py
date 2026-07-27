import httpx
import pytest

from harness.tools.base import ToolRegistry, ToolExecutor
from harness.tools.builtins.http_tool import HttpRequestTool
from harness.types import ToolCall


def _factory(handler):
    return lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)


def _public(host):
    return ["93.184.216.34"]


async def test_success_returns_body():
    def handler(req):
        return httpx.Response(200, text="hello world")
    tool = HttpRequestTool([], True, 5.0, 1000, 3, client_factory=_factory(handler), resolve=_public)
    out = await tool.run(tool.Params(url="http://example.com/"))
    assert "200" in out and "hello world" in out


async def test_ssrf_blocked_is_error():
    def handler(req):
        return httpx.Response(200, text="secret")
    tool = HttpRequestTool([], True, 5.0, 1000, 3,
                           client_factory=_factory(handler), resolve=lambda h: ["127.0.0.1"])
    reg = ToolRegistry(); reg.register(tool)
    ex = ToolExecutor(reg)
    r = await ex.execute(ToolCall(id="c1", name="http_request",
                                  arguments={"url": "http://internal/"}))
    assert r.is_error is True


async def test_response_truncated():
    def handler(req):
        return httpx.Response(200, text="A" * 5000)
    tool = HttpRequestTool([], True, 5.0, 100, 3, client_factory=_factory(handler), resolve=_public)
    out = await tool.run(tool.Params(url="http://example.com/"))
    assert "…(已截断)" in out


_ARTICLE = (
    "<html><head><title>光合作用</title></head><body>"
    "<nav>首页 关于 登录 注册</nav>"
    "<article><h1>光合作用的原理</h1>"
    "<p>光合作用是绿色植物利用光能，把二氧化碳和水转化为储存能量的有机物，并释放氧气的过程。"
    "这一过程主要发生在叶绿体中，是地球上几乎所有生命能量的最终来源，对维持大气中氧气与二氧化碳的平衡至关重要。</p>"
    "</article><footer>版权所有 © 2026</footer></body></html>"
)


async def test_html_returns_parsed_title_and_text():
    def handler(req):
        return httpx.Response(200, text=_ARTICLE, headers={"content-type": "text/html; charset=utf-8"})
    tool = HttpRequestTool([], True, 5.0, 5_000_000, 3, client_factory=_factory(handler), resolve=_public)
    out = await tool.run(tool.Params(url="http://example.com/"))
    assert "标题：光合作用" in out
    assert "光合作用是绿色植物" in out
    assert "<html>" not in out and "<nav>" not in out   # 不再倒原始标签墙
    assert "登录 注册" not in out                          # 样板被去掉


async def test_raw_flag_returns_original_html():
    def handler(req):
        return httpx.Response(200, text=_ARTICLE, headers={"content-type": "text/html"})
    tool = HttpRequestTool([], True, 5.0, 5_000_000, 3, client_factory=_factory(handler), resolve=_public)
    out = await tool.run(tool.Params(url="http://example.com/", raw=True))
    assert "<nav>" in out                                  # raw=true 原样返回


async def test_json_passthrough():
    def handler(req):
        return httpx.Response(200, text='{"ok": true, "n": 1}',
                              headers={"content-type": "application/json"})
    tool = HttpRequestTool([], True, 5.0, 5_000_000, 3, client_factory=_factory(handler), resolve=_public)
    out = await tool.run(tool.Params(url="http://example.com/api"))
    assert '{"ok": true, "n": 1}' in out
    assert "标题：" not in out


async def test_redirect_to_internal_blocked():
    def handler(req):
        if req.url.host == "example.com":
            return httpx.Response(302, headers={"location": "http://internal/"})
        return httpx.Response(200, text="internal")
    # example.com 公网、internal 解析到内网
    def resolve(host):
        return ["93.184.216.34"] if host == "example.com" else ["127.0.0.1"]
    tool = HttpRequestTool([], True, 5.0, 1000, 3, client_factory=_factory(handler), resolve=resolve)
    reg = ToolRegistry(); reg.register(tool)
    ex = ToolExecutor(reg)
    r = await ex.execute(ToolCall(id="c1", name="http_request",
                                  arguments={"url": "http://example.com/"}))
    assert r.is_error is True   # 第二跳内网被策略拦截


# ---------- 浏览器自动兜底 ----------

from harness.tools.builtins.http_tool import looks_blocked  # noqa: E402


def test_looks_blocked_heuristics():
    assert looks_blocked(403, "") is True
    assert looks_blocked(429, "") is True
    assert looks_blocked(200, "Just a moment... Cloudflare") is True
    assert looks_blocked(200, "请开启JavaScript后重试") is True
    assert looks_blocked(200, "正常正文内容") is False


def _fallback(record, text="浏览器抓到的正文", fail=False):
    async def fn(url):
        record.append(url)
        if fail:
            raise RuntimeError("ModuleNotFoundError: No module named 'playwright'")
        return text
    return fn


async def test_fallback_on_network_error():
    def handler(req):
        raise httpx.ConnectError("boom")
    calls = []
    tool = HttpRequestTool([], True, 5.0, 1000, 3, client_factory=_factory(handler),
                           resolve=_public, browser_fallback=_fallback(calls))
    out = await tool.run(tool.Params(url="http://example.com/"))
    assert calls == ["http://example.com/"]
    assert "已自动改用浏览器抓取" in out and "浏览器抓到的正文" in out


async def test_fallback_on_blocked_status():
    def handler(req):
        return httpx.Response(403, text="Access Denied")
    calls = []
    tool = HttpRequestTool([], True, 5.0, 2000, 3, client_factory=_factory(handler),
                           resolve=_public, browser_fallback=_fallback(calls))
    out = await tool.run(tool.Params(url="http://example.com/"))
    assert calls and "浏览器抓到的正文" in out


async def test_policy_error_does_not_fallback():
    def handler(req):
        return httpx.Response(200, text="secret")
    calls = []
    tool = HttpRequestTool([], True, 5.0, 1000, 3, client_factory=_factory(handler),
                           resolve=lambda h: ["127.0.0.1"], browser_fallback=_fallback(calls))
    reg = ToolRegistry(); reg.register(tool)
    r = await ToolExecutor(reg).execute(
        ToolCall(id="c1", name="http_request", arguments={"url": "http://internal/"}))
    assert r.is_error is True
    assert calls == []          # 安全拦截不触发浏览器兜底


def handler_403(req):
    return httpx.Response(403, text="Access Denied")


async def test_blocked_but_fallback_fails_says_so_plainly():
    """兜底失败时仍要试过兜底、且不隐瞒被拦（本测试的原意），但不再把拦截页当正文交出去。

    原先退回 render_http_result：状态码是带出来了，可整体形态与一次正常抓取无异，
    模型容易把「Access Denied」当页面内容继续用。现在明说抓取失败并给出下一步。
    """
    calls = []
    tool = HttpRequestTool([], True, 5.0, 2000, 3, client_factory=_factory(handler_403),
                           resolve=_public, browser_fallback=_fallback(calls, fail=True))
    out = await tool.run(tool.Params(url="http://example.com/"))
    assert calls                                     # 尝试过兜底（原意保留）
    assert "403" in out                              # 不隐瞒具体状态（原意保留）
    assert "抓取失败" in out and "不要引用" in out    # 但要明说不可用


async def test_normal_page_no_fallback():
    def handler(req):
        return httpx.Response(200, text="<html><body>正文很充实的内容</body></html>")
    calls = []
    tool = HttpRequestTool([], True, 5.0, 2000, 3, client_factory=_factory(handler),
                           resolve=_public, browser_fallback=_fallback(calls))
    out = await tool.run(tool.Params(url="http://example.com/"))
    assert calls == []          # 正常页面不触发兜底
    assert "200" in out


# ---- User-Agent：空 UA 会被不少站点直接 403，故默认必须带一个 ----

_UA = "Mozilla/5.0 (compatible; AI-Learning-Helper/1.0; +harness)"


async def test_default_user_agent_is_sent():
    seen = {}

    def handler(req):
        seen["ua"] = req.headers.get("user-agent")
        return httpx.Response(200, text="ok")
    tool = HttpRequestTool([], True, 5.0, 1000, 3, client_factory=_factory(handler),
                           resolve=_public, user_agent=_UA)
    await tool.run(tool.Params(url="http://example.com/"))
    assert seen["ua"] == _UA


async def test_explicit_user_agent_wins_over_default():
    seen = {}

    def handler(req):
        seen["ua"] = req.headers.get("user-agent")
        return httpx.Response(200, text="ok")
    tool = HttpRequestTool([], True, 5.0, 1000, 3, client_factory=_factory(handler),
                           resolve=_public, user_agent=_UA)
    await tool.run(tool.Params(url="http://example.com/",
                               headers={"user-agent": "MyBot/9"}))   # 大小写不敏感
    assert seen["ua"] == "MyBot/9"


async def test_other_headers_survive_ua_merge():
    seen = {}

    def handler(req):
        seen["auth"] = req.headers.get("authorization")
        seen["ua"] = req.headers.get("user-agent")
        return httpx.Response(200, text="ok")
    tool = HttpRequestTool([], True, 5.0, 1000, 3, client_factory=_factory(handler),
                           resolve=_public, user_agent=_UA)
    await tool.run(tool.Params(url="http://example.com/", headers={"Authorization": "Bearer t"}))
    assert seen["auth"] == "Bearer t" and seen["ua"] == _UA


async def test_ua_merge_does_not_mutate_caller_headers():
    def handler(req):
        return httpx.Response(200, text="ok")
    tool = HttpRequestTool([], True, 5.0, 1000, 3, client_factory=_factory(handler),
                           resolve=_public, user_agent=_UA)
    headers = {"X-K": "v"}
    await tool.run(tool.Params(url="http://example.com/", headers=headers))
    assert headers == {"X-K": "v"}          # 调用方传进来的 dict 不该被就地改写


async def test_empty_user_agent_config_sends_none():
    # 显式设空 → 退回旧行为（不发 UA），逃生口
    seen = {}

    def handler(req):
        seen["ua"] = req.headers.get("user-agent")
        return httpx.Response(200, text="ok")
    tool = HttpRequestTool([], True, 5.0, 1000, 3, client_factory=_factory(handler),
                           resolve=_public, user_agent="")
    await tool.run(tool.Params(url="http://example.com/"))
    assert seen["ua"] is None or "python-httpx" in seen["ua"].lower()


# ---------- 防抓页不得冒充正文 ----------

def _blocked_tool(status, body, *, fallback=None):
    """构造一个恒返回指定响应的 HttpRequestTool（走真实 client_factory，不打网络）。"""
    def handler(req):
        return httpx.Response(status, text=body)
    return HttpRequestTool([], True, 5.0, 200_000, 3, client_factory=_factory(handler),
                           resolve=_public, browser_fallback=fallback)


async def test_block_status_without_browser_refuses_to_pass_page_off_as_content():
    """回归：浏览器兜底不可用时，拦截页曾被原样当正文返回。

    模型拿到的是格式完全正常的「标题+正文」，于是照着人机验证页总结。
    looks_blocked 已经判定它是拦截页，这个结论不能丢。
    """
    t = _blocked_tool(403, "Access Denied")
    out = await t.run(t.Params(url="http://example.com/a"))
    assert "抓取失败" in out and "不要引用" in out
    assert "换一个可访问的来源" in out


async def test_keyword_only_hit_keeps_content_but_warns():
    """仅正文关键词命中（HTTP 200）→ 可能是误判，内容照给但把疑点摆前面。

    一篇正经讨论「人机验证」的文章会命中关键词。若一律判死，误判的代价会从
    「白试一次浏览器、照样返回内容」恶化成「内容被整个吞掉」，比不修还糟。
    """
    article = "本文讨论人机验证的实现原理。" * 20
    t = _blocked_tool(200, article)
    out = await t.run(t.Params(url="http://example.com/a"))
    assert "疑似" in out                       # 有警示
    assert "人机验证的实现原理" in out          # 但正文没丢


async def test_browser_fallback_still_wins_when_available():
    """有浏览器兜底时仍走兜底，不受本次改动影响。"""
    async def _fb(_url):
        return "浏览器抓到的真实正文"
    t = _blocked_tool(403, "Access Denied", fallback=_fb)
    out = await t.run(t.Params(url="http://example.com/a"))
    assert "浏览器抓到的真实正文" in out
    assert "抓取失败" not in out


def test_baidu_challenge_wording_is_recognised():
    """百度系挑战页写的是「百度安全验证」，与信号表里既有的「人机验证」不是同一个词。"""
    from harness.tools.builtins.http_tool import looks_blocked
    assert looks_blocked(200, "<title>百度安全验证</title>请完成验证")
    assert not looks_blocked(200, "一篇讲人工智能的普通文章")
