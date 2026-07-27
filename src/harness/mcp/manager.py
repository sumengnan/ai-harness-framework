"""MCP 连接管理器：读清单 → 连接每个 server → 暴露远程工具为 McpTool。

生命周期要点（务必遵守）：stdio_client / streamablehttp_client / ClientSession 都是
async context manager，anyio 要求**同一 task 内 enter+exit**；跨 startup/shutdown 存
AsyncExitStack 会触发 "cancel scope in a different task"。故每个 server 用一个常驻后台
task：task 内 `async with` 全程持有连接，靠 stop Event 收尾，ready Event 通知启动完成。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from pydantic import BaseModel, ValidationError

from .tool import McpTool

logger = logging.getLogger("harness.mcp")


class _ServerCfg(BaseModel):
    name: str
    enabled: bool = True
    transport: str = "stdio"          # "stdio" | "http"
    # stdio
    command: str | None = None
    args: list[str] = []
    env: dict[str, str] = {}
    # http (streamable-http)
    url: str | None = None
    headers: dict[str, str] = {}
    timeout: float = 30.0


def _expand(value: str) -> str:
    return os.path.expandvars(value) if isinstance(value, str) else value


def load_server_configs(path: str) -> list[_ServerCfg]:
    """读 MCP server 清单（默认 mcp/mcp_servers.json）。文件不存在视为空列表；非法条目跳过并 warning。"""
    if not path or not os.path.isfile(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("读取 MCP 配置 %s 失败：%s", path, e)
        return []
    out: list[_ServerCfg] = []
    for i, item in enumerate(raw.get("servers", []) if isinstance(raw, dict) else []):
        try:
            out.append(_ServerCfg.model_validate(item))
        except ValidationError as e:
            logger.warning("MCP 配置第 %d 条非法，已跳过：%s", i, e)
    return out


class _ServerConn:
    """单个 server 的常驻连接：一个后台 task 内全程持有 ClientSession。"""

    def __init__(self, manager: "MCPManager", cfg: _ServerCfg) -> None:
        self._manager = manager
        self.cfg = cfg
        self.session: Any = None
        self.tools: list[McpTool] = []
        self.error: Exception | None = None
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None

    def _transport(self):
        if self.cfg.transport == "http":
            from mcp.client.streamable_http import streamablehttp_client
            if not self.cfg.url:
                raise ValueError("http 传输缺少 url")
            headers = {k: _expand(v) for k, v in self.cfg.headers.items()}
            return streamablehttp_client(
                _expand(self.cfg.url), headers=headers or None, timeout=self.cfg.timeout)
        # stdio
        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client
        if not self.cfg.command:
            raise ValueError("stdio 传输缺少 command")
        params = StdioServerParameters(
            command=self.cfg.command,
            args=[_expand(a) for a in self.cfg.args],
            env={**os.environ, **{k: _expand(v) for k, v in self.cfg.env.items()}})
        return stdio_client(params)

    async def _run(self) -> None:
        from mcp import ClientSession
        try:
            async with self._transport() as streams:
                # stdio 返回 (read, write)；streamable-http 返回三元组，第三个（get_session_id）丢弃
                read, write = streams[0], streams[1]
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    resp = await session.list_tools()
                    self.session = session
                    self.tools = [
                        McpTool(self._manager, self.cfg.name, t.name,
                                t.description or "", t.inputSchema)
                        for t in resp.tools]
                    logger.info("MCP server '%s' 已连接，%d 个工具", self.cfg.name, len(self.tools))
                    self._ready.set()
                    await self._stop.wait()     # 常驻，直到 close()
        except Exception as e:  # 连接/初始化失败：记录，不抛出（不拖垮启动）
            self.error = e
            logger.warning("MCP server '%s' 连接失败，已跳过：%s", self.cfg.name, e)
        finally:
            self.session = None
            self._ready.set()                   # 失败也放行 start() 的等待

    async def start(self, timeout: float) -> None:
        self._task = asyncio.create_task(self._run())
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("MCP server '%s' 连接超时（%.0fs），已跳过", self.cfg.name, timeout)

    async def close(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await self._task
            except Exception:  # 收尾异常不外泄
                pass


class MCPManager:
    """按配置连接一组 MCP server，把远程工具暴露为 McpTool，供注册进 registry。"""

    def __init__(self, config) -> None:
        self._config_path = getattr(config, "mcp_config_path", "mcp/mcp_servers.json")
        self._timeout = getattr(config, "mcp_connect_timeout", 15.0)
        self._conns: dict[str, _ServerConn] = {}

    async def start(self) -> None:
        for cfg in load_server_configs(self._config_path):
            if not cfg.enabled:
                continue
            if cfg.name in self._conns:
                logger.warning("MCP server 名 '%s' 重复，跳过后者", cfg.name)
                continue
            conn = _ServerConn(self, cfg)
            self._conns[cfg.name] = conn
            await conn.start(self._timeout)

    async def call(self, server_name: str, tool_name: str, args: dict):
        conn = self._conns.get(server_name)
        if conn is None or conn.session is None:
            raise RuntimeError(f"MCP server '{server_name}' 未连接")
        return await conn.session.call_tool(tool_name, args)

    def tools(self) -> list[McpTool]:
        out: list[McpTool] = []
        for conn in self._conns.values():
            out.extend(conn.tools)
        return out

    def tool_names(self) -> list[str]:
        return [t.name for t in self.tools()]

    def status(self) -> list[dict]:
        return [
            {"name": c.cfg.name, "transport": c.cfg.transport,
             "connected": c.session is not None,
             "tools": [t.name for t in c.tools],
             "error": str(c.error) if c.error else None}
            for c in self._conns.values()]

    async def close(self) -> None:
        await asyncio.gather(*(c.close() for c in self._conns.values()),
                             return_exceptions=True)
        self._conns.clear()

    async def reload(self) -> list[McpTool]:
        """关掉全部连接、重读清单、重连。返回新的工具列表。"""
        await self.close()
        await self.start()
        return self.tools()
