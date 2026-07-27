"""示范用内置 MCP server：暴露若干**无状态、无 per-user 依赖**的工具。

用来端到端证明 harness 的 MCP 客户端链路可用（配置解析 → 连接 → 工具适配 → 调用）。
有状态 / per-user 的工具不适合跨进程迁移，应留在消费方进程内作为本地 Tool。

启动：
    python -m fixtures.example_mcp_server            # 默认 stdio
    python -m fixtures.example_mcp_server --http     # streamable-http（默认 127.0.0.1:9100）

（需 PYTHONPATH 含 src/ 与 tests/，见 tests/test_mcp.py 的 _stdio_manifest。）
"""
from __future__ import annotations

import argparse

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("example")


@mcp.tool()
def calc(expression: str) -> str:
    """对纯数字做算术计算，支持 + - * / ** % 和括号（如 3*(4+5)、2**10）。

    只接受数值表达式：不支持日期/时间运算（如 "2026-07-19 + 2"）、变量、函数或文本。
    """
    # 复用现成的受限 AST 求值（绝不 eval，天然安全）
    from harness.tools.builtins.calculator import safe_eval
    return str(safe_eval(expression))


@mcp.tool()
def now() -> str:
    """返回当前 UTC 时间的 ISO8601 字符串。"""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", action="store_true", help="用 streamable-http 传输启动")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9100)
    args = parser.parse_args()
    if args.http:
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.run(transport="streamable-http")
    else:
        mcp.run()   # 默认 stdio


if __name__ == "__main__":
    main()
