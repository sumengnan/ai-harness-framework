# src/harness/tools/builtins/fs_tools.py
from __future__ import annotations

from pydantic import BaseModel

from ..base import Tool
from ...sandbox.base import Sandbox
from ._sandbox_util import truncate

# 每种语言在各自独立容器里执行、工作区互不可见。文件工具用 language 参数选落到哪个容器：
# 带 language（如 "python"）→ 落该语言容器，随后 run_python 就能读到；缺省 → 落 shell 容器
# （run_shell 那个），run_python 等看不到。跨语言之间文件不共享。
_LANG_HINT = ("可选 language 指定落到哪个语言容器（如 \"python\"/\"node\"/\"java\"）——"
              "带上后对应的 run_<语言> 才能读到该文件；缺省落 shell 容器（run_shell 用）。"
              "各语言容器工作区相互独立、不共享。")


class WriteFileTool(Tool):
    name = "write_file"
    description = "在沙箱工作区写入文件（路径限工作区内）。" + _LANG_HINT

    class Params(BaseModel):
        path: str
        content: str
        language: str | None = None

    def __init__(self, sandbox: Sandbox) -> None:
        self._sandbox = sandbox

    async def run(self, params: "WriteFileTool.Params") -> str:
        box = await self._sandbox.for_language(params.language)   # SandboxError→is_error
        await box.write_file(params.path, params.content)
        where = f"（{params.language} 容器）" if params.language else "（shell 容器）"
        return f"已写入 {params.path}{where}（{len(params.content)} 字符）。"


class ReadFileTool(Tool):
    name = "read_file"
    description = "读取沙箱工作区的文件。" + _LANG_HINT

    class Params(BaseModel):
        path: str
        language: str | None = None

    def __init__(self, sandbox: Sandbox, max_chars: int = 8000) -> None:
        self._sandbox = sandbox
        self._max_chars = max_chars

    async def run(self, params: "ReadFileTool.Params") -> str:
        box = await self._sandbox.for_language(params.language)
        content = await box.read_file(params.path)   # SandboxError/FileNotFound→is_error
        return truncate(content, self._max_chars)


class ListFilesTool(Tool):
    name = "list_files"
    description = "列出沙箱工作区目录内容。" + _LANG_HINT

    class Params(BaseModel):
        path: str = "."
        language: str | None = None

    def __init__(self, sandbox: Sandbox) -> None:
        self._sandbox = sandbox

    async def run(self, params: "ListFilesTool.Params") -> str:
        box = await self._sandbox.for_language(params.language)
        files = await box.list_files(params.path)
        return "\n".join(files) if files else "（空目录）"
