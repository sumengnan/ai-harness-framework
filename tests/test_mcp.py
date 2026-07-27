"""MCP 客户端集成测试：配置解析、McpTool 适配、stdio 端到端。"""
from __future__ import annotations

import json
import os
import sys

import pytest

import harness
from harness.mcp.manager import MCPManager, load_server_configs
from harness.mcp.tool import McpTool, _content_to_text
from harness.tools.base import ToolExecutor, ToolRegistry
from harness.types import ToolCall

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(harness.__file__)))  # .../src
_ROOT = os.path.dirname(_SRC)                                              # 仓库根
_TESTS = os.path.dirname(os.path.abspath(__file__))                        # tests/（含 fixtures 包）


class _Cfg:
    def __init__(self, path: str, timeout: float = 20.0) -> None:
        self.mcp_config_path = path
        self.mcp_connect_timeout = timeout


# ---------- 配置解析 ----------

def test_load_missing_file_returns_empty(tmp_path):
    assert load_server_configs(str(tmp_path / "nope.json")) == []


def test_load_skips_invalid_entries(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"servers": [
        {"name": "ok", "command": "python"},          # 合法
        {"transport": "stdio"},                        # 非法：缺 name，应跳过
    ]}), encoding="utf-8")
    cfgs = load_server_configs(str(p))
    assert [c.name for c in cfgs] == ["ok"]


# ---------- McpTool 适配（用假 manager，不起子进程） ----------

class _FakeManager:
    def __init__(self, result):
        self._result = result
        self.calls = []

    async def call(self, server, tool, args):
        self.calls.append((server, tool, args))
        return self._result


def _tool_result(text=None, is_error=False, structured=None):
    from mcp.types import CallToolResult, TextContent
    content = [TextContent(type="text", text=text)] if text is not None else []
    return CallToolResult(content=content, isError=is_error, structuredContent=structured)


def test_mcp_tool_name_and_schema_passthrough():
    schema = {"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"]}
    t = McpTool(_FakeManager(_tool_result("ok")), "srv", "add", "加法", schema)
    assert t.name == "mcp__srv__add"                 # 双下划线前缀防撞名
    fn = t.schema()["function"]
    assert fn["name"] == "mcp__srv__add"
    assert fn["description"] == "加法"
    assert fn["parameters"] == schema                # inputSchema 原样透传


async def test_mcp_tool_run_passes_args_and_returns_text():
    mgr = _FakeManager(_tool_result("625"))
    reg = ToolRegistry()
    reg.register(McpTool(mgr, "srv", "calc", "算", {"type": "object"}))
    ex = ToolExecutor(reg, max_chars=8000)
    r = await ex.execute(ToolCall(id="1", name="mcp__srv__calc", arguments={"expression": "(2+3)**4"}))
    assert r.content == "625" and not r.is_error
    # 透传模型把任意参数原样交给远端 call
    assert mgr.calls == [("srv", "calc", {"expression": "(2+3)**4"})]


async def test_mcp_tool_error_marks_is_error():
    reg = ToolRegistry()
    reg.register(McpTool(_FakeManager(_tool_result("boom", is_error=True)), "srv", "f", "", {}))
    ex = ToolExecutor(reg)
    r = await ex.execute(ToolCall(id="1", name="mcp__srv__f", arguments={}))
    assert r.is_error and "boom" in r.content


def test_content_to_text_falls_back_to_structured():
    assert _content_to_text(_tool_result(structured={"a": 1})) == '{"a": 1}'


# ---------- stdio 端到端（起内置示范 server 子进程） ----------

def _stdio_manifest(tmp_path):
    p = tmp_path / "mcp.json"
    p.write_text(json.dumps({"servers": [{
        "name": "example", "enabled": True, "transport": "stdio",
        "command": sys.executable, "args": ["-m", "fixtures.example_mcp_server"],
        # 保证子进程导入与本进程同一份 harness，并能找到 fixtures 夹具
        "env": {"PYTHONPATH": os.pathsep.join([_SRC, _TESTS])},
    }]}), encoding="utf-8")
    return str(p)


async def test_stdio_end_to_end(tmp_path):
    mgr = MCPManager(_Cfg(_stdio_manifest(tmp_path)))
    await mgr.start()
    try:
        names = mgr.tool_names()
        assert "mcp__example__calc" in names
        assert "mcp__example__now" in names
        reg = ToolRegistry()
        for t in mgr.tools():
            reg.register(t)
        ex = ToolExecutor(reg, max_chars=8000)
        r = await ex.execute(ToolCall(
            id="1", name="mcp__example__calc", arguments={"expression": "(2+3)**4"}))
        assert r.content == "625" and not r.is_error
    finally:
        await mgr.close()


async def test_start_tolerates_bad_server(tmp_path):
    p = tmp_path / "mcp.json"
    p.write_text(json.dumps({"servers": [
        {"name": "bad", "enabled": True, "transport": "stdio", "command": "no_such_cmd_xyz"},
        {"name": "good", "enabled": True, "transport": "stdio", "command": sys.executable,
         "args": ["-m", "fixtures.example_mcp_server"],
         "env": {"PYTHONPATH": os.pathsep.join([_SRC, _TESTS])}},
    ]}), encoding="utf-8")
    mgr = MCPManager(_Cfg(p and str(p)))
    await mgr.start()   # 不应抛异常
    try:
        st = {s["name"]: s["connected"] for s in mgr.status()}
        assert st == {"bad": False, "good": True}   # 坏的跳过、好的连上
        assert "mcp__good__calc" in mgr.tool_names()
    finally:
        await mgr.close()
