# src/harness/sandbox/local.py
from __future__ import annotations

import asyncio
import os
import shutil
import tempfile

from .base import ExecResult, resolve_in_workspace


class LocalSandbox:
    """本地目录 + 子进程。仅测试/离线开发用——不是安全边界。

    workspace=None：自建临时目录，close() 时删除（默认，行为不变）。
    workspace=路径：接管既有目录，close() **不删**——用于在 git worktree
    这类由调用方管理生命周期的目录上工作。
    """

    def __init__(self, workspace: str | None = None) -> None:
        self._adopted = workspace is not None
        self.workspace = workspace or ""
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        if not self._adopted:
            self.workspace = tempfile.mkdtemp(prefix="harness_sbx_")
        self._started = True

    async def close(self) -> None:
        # 接管的目录归调用方所有，绝不删除
        if (not self._adopted and self._started
                and self.workspace and os.path.isdir(self.workspace)):
            shutil.rmtree(self.workspace, ignore_errors=True)
        self._started = False

    async def for_language(self, language: str | None = None,
                           version: str | None = None) -> "LocalSandbox":
        # 本地沙箱无镜像/隔离概念：所有语言共用这一个本地目录（仅测试/离线开发用）
        await self.start()
        return self

    async def exec(self, command: list[str], timeout: float,
                   *, quiet: bool = False) -> ExecResult:
        await self.start()   # 本地沙箱本就不发 Progress，quiet 仅为接口一致
        proc = await asyncio.create_subprocess_exec(
            *command, cwd=self.workspace,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return ExecResult(out.decode(errors="replace"), err.decode(errors="replace"),
                              proc.returncode, timed_out=False)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return ExecResult("", f"超时（>{timeout}s）被终止", -1, timed_out=True)

    async def write_file(self, path: str, content: str) -> None:
        await self.start()
        real = resolve_in_workspace(self.workspace, path)
        os.makedirs(os.path.dirname(real), exist_ok=True)
        with open(real, "w") as f:
            f.write(content)

    async def write_bytes(self, path: str, data: bytes) -> None:
        await self.start()
        real = resolve_in_workspace(self.workspace, path)
        os.makedirs(os.path.dirname(real), exist_ok=True)
        with open(real, "wb") as f:
            f.write(data)

    async def read_file(self, path: str) -> str:
        await self.start()
        real = resolve_in_workspace(self.workspace, path)
        with open(real) as f:
            return f.read()

    async def list_files(self, path: str = ".") -> list[str]:
        await self.start()
        real = resolve_in_workspace(self.workspace, path)
        return sorted(os.listdir(real))
