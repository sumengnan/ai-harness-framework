from harness.loop.agent_loop import AgentLoop
from harness.context.manager import ContextManager
from harness.tools.base import ToolRegistry
from harness.browser.fake import FakeBrowser
from harness.tools.builtins.browse_tool import BrowseTool
from harness.llm.base import StreamChunk, ToolCallDelta
from harness.events import ToolFinished, RunFinished

PAGE = """
<html><head><title>Py教程</title></head><body>
<nav>菜单</nav>
<article><p>Python 是一门通用编程语言，语法简洁、生态丰富，广泛用于数据分析、Web 开发与人工智能等领域，
非常适合初学者作为第一门编程语言来学习和实践。</p></article>
<footer>脚注</footer></body></html>
"""


async def test_agent_browses_page(make_mock, text_turn):
    fb = FakeBrowser({"http://example.com/py": ("Py教程", PAGE)})
    reg = ToolRegistry()
    reg.register(BrowseTool(fb, allowed_domains=[], block_private=True, timeout=5,
                            wait_until="load", max_chars=8000,
                            resolve=lambda h: ["93.184.216.34"]))
    browse_turn = [
        StreamChunk(type="tool_call", tool_call_delta=ToolCallDelta(
            index=0, id="c1", name="browse",
            arguments='{"url": "http://example.com/py"}')),
        StreamChunk(type="done"),
    ]
    loop = AgentLoop(client=make_mock([browse_turn, text_turn("据网页，Python 适合初学者")]),
                     registry=reg, context=ContextManager(system_prompt="s"),
                     max_steps=5, run_id_factory=lambda: "r1")
    events = [e async for e in loop.run("介绍下 Python")]
    finished = [e for e in events if isinstance(e, ToolFinished)]
    assert "通用编程语言" in finished[0].result.content
    assert finished[0].result.is_error is False
    assert isinstance(events[-1], RunFinished)
