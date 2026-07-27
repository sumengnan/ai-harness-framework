"""把远程 MCP 工具包装成本地 Tool，零改动接入现有 agent loop / ToolExecutor。

核心适配点：`Tool.schema()`（harness/tools/base.py）依赖 `Params.model_json_schema()`，
而 MCP 只给现成 JSON Schema（`inputSchema`）；`ToolExecutor` 又会
`Params.model_validate(args)` 再 `run(params)`。所以：
- override `schema()` 直接透传 MCP 的 inputSchema；
- `Params` 用 extra="allow" 的透传模型让校验放行，`model_dump()` 原样带出参数。
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from ..tools.base import Tool, ToolError

if TYPE_CHECKING:  # 避免运行时强依赖 mcp 类型
    from mcp.types import CallToolResult

    from .manager import MCPManager


class _PassthroughParams(BaseModel):
    """透传模型：收下任意字段，model_dump() 原样交给 call_tool。

    真正的入参约束由 override 的 schema() 交给模型侧，并在远端 server 再校一次。
    不从 JSON Schema 反造 pydantic 模型（$ref/anyOf/嵌套易碎，收益低）。
    """

    model_config = ConfigDict(extra="allow")


def _content_to_text(result: "CallToolResult") -> str:
    """把 CallToolResult 拼成字符串喂回模型：优先文本块，退回 structuredContent。"""
    parts: list[str] = []
    for block in result.content or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
        else:  # image/audio/embedded resource 等非文本块：给占位说明
            parts.append(f"[非文本内容: {getattr(block, 'type', 'unknown')}]")
    if parts:
        return "\n".join(parts)
    if result.structuredContent is not None:
        return json.dumps(result.structuredContent, ensure_ascii=False)
    return ""


class McpTool(Tool):
    """一个远程 MCP 工具的本地代理。名字加 mcp__<server>__<tool> 前缀防撞名。"""

    Params = _PassthroughParams

    def __init__(self, manager: "MCPManager", server_name: str, remote_name: str,
                 description: str, input_schema: dict) -> None:
        self._manager = manager
        self._server_name = server_name
        self._remote_name = remote_name          # 远端真实工具名（call_tool 用）
        # OpenAI function name 只允许 [a-zA-Z0-9_-]，故用双下划线而非点分隔
        self.name = f"mcp__{server_name}__{remote_name}"
        self.description = description or ""
        self._input_schema = input_schema or {"type": "object", "properties": {}}

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._input_schema,   # 直接透传 MCP inputSchema
            },
        }

    async def run(self, params: BaseModel) -> str:
        args = params.model_dump()
        result = await self._manager.call(self._server_name, self._remote_name, args)
        text = _content_to_text(result)
        if result.isError:
            # 复用现有语义：内容原样回传、标 is_error，让模型自纠正
            raise ToolError(text or "MCP 工具返回错误")
        return text
