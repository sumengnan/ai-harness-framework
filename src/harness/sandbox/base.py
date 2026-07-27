# src/harness/sandbox/base.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False


class SandboxError(Exception):
    ...


def resolve_in_workspace(workspace: str, path: str) -> str:
    """把 path 规约到 workspace 内的绝对路径；逃逸（../、绝对路径、符号链接）抛 SandboxError。"""
    workspace_real = os.path.realpath(workspace)
    candidate = path if os.path.isabs(path) else os.path.join(workspace_real, path)
    real = os.path.realpath(candidate)
    if real != workspace_real and not real.startswith(workspace_real + os.sep):
        raise SandboxError(f"路径逃逸工作区：{path}")
    return real


@runtime_checkable
class Sandbox(Protocol):
    workspace: str

    async def start(self) -> None: ...
    async def close(self) -> None: ...
    # 取到「执行该语言代码」的已启动容器；language=None/未知 → 缺省容器（shell）。
    # 单容器实现（Docker/Local 直连）返回自身；会话代理据 (会话,语言) 路由到对应容器。
    async def for_language(self, language: str | None = None,
                           version: str | None = None) -> "Sandbox": ...
    async def exec(self, command: list[str], timeout: float,
                   *, quiet: bool = False) -> ExecResult: ...
    async def write_file(self, path: str, content: str) -> None: ...
    async def write_bytes(self, path: str, data: bytes) -> None: ...
    async def read_file(self, path: str) -> str: ...
    async def list_files(self, path: str = ".") -> list[str]: ...
