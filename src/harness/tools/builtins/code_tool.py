# src/harness/tools/builtins/code_tool.py
from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from ..base import Tool, ToolError
from ...sandbox.base import Sandbox
from ...types import ToolOutput
from ._sandbox_util import format_exec


@dataclass
class LangSpec:
    language: str            # 语言 key（对应 config.sandbox_lang_images 的键，用于路由到语言容器）
    filename: str            # 源码写入的文件名
    argv: list[str] | None   # 直接执行的命令；None 表示用 shell 编译+运行
    shell: str | None = None # 需编译时的 sh -c 脚本（如 java 先 javac 再 java）


async def _run_code(sandbox: Sandbox, spec: LangSpec, code: str, version: str | None,
                    timeout: float, max_chars: int) -> str:
    """执行某语言代码；非零退出/超时 → ToolError。

    经 for_language(语言[,版本]) 取到该语言的容器（会话代理据此起/复用对应语言容器；
    直连 Docker/Local 时返回自身），在其中 write_file + exec。显式 version 无对应镜像
    → SandboxError（is_error，让模型换版本）。
    """
    box = await sandbox.for_language(spec.language, version)   # 版本无镜像等→SandboxError（尚无容器/镜像）
    meta = {"image": box.image} if getattr(box, "image", None) else None   # 供前端标注用的镜像
    cmd = spec.argv if spec.argv is not None else ["sh", "-c", spec.shell]
    try:
        await box.write_file(spec.filename, code)
        res = await box.exec(cmd, timeout)
    except ToolError:
        raise
    except Exception as e:     # 容器级异常（SandboxError 等）也带上镜像 meta，保证成功失败都显示镜像
        raise ToolError(str(e), meta=meta)
    out = format_exec(res, max_chars)
    if res.exit_code != 0 or res.timed_out:     # 非零退出/超时 → 标记失败（失败也带镜像 meta）
        raise ToolError(out, meta=meta)
    return ToolOutput(text=out, meta=meta) if meta else out


class _CodeTool(Tool):
    """按 LangSpec 在沙箱内跑某种语言代码的通用基类。"""
    spec: LangSpec

    class Params(BaseModel):
        code: str
        version: str | None = None   # 语言版本（如 java 的 8/17/21）；空=用默认版本镜像

    def __init__(self, sandbox: Sandbox, timeout: float = 30.0, max_chars: int = 8000) -> None:
        self._sandbox = sandbox
        self._timeout = timeout
        self._max_chars = max_chars

    async def run(self, params: "_CodeTool.Params") -> str:
        return await _run_code(self._sandbox, self.spec, params.code, params.version,
                               self._timeout, self._max_chars)


class RunPythonTool(_CodeTool):
    name = "run_python"
    description = "在沙箱内执行 Python 代码，返回输出。"
    spec = LangSpec("python", "_run.py", ["python3", "_run.py"])

    def __init__(self, sandbox: Sandbox, timeout: float = 30.0,
                 max_chars: int = 8000, python_cmd: str = "python3") -> None:
        super().__init__(sandbox, timeout, max_chars)
        # 兼容旧签名：允许自定义 python 解释器命令
        if python_cmd != "python3":
            self.spec = LangSpec("python", "_run.py", [python_cmd, "_run.py"])


class RunNodeTool(_CodeTool):
    name = "run_node"
    description = "在沙箱内执行 Node.js 代码，返回输出。"
    spec = LangSpec("node", "_run.js", ["node", "_run.js"])


class RunJavaTool(_CodeTool):
    name = "run_java"
    description = ("在沙箱内执行 Java 代码，返回输出。"
                   "入口类必须命名为 public class Main（含 public static void main）。"
                   "可选 version 指定 JDK 版本（如 8/11/17/21，取决于服务端 sandbox_lang_images 配置），"
                   "空则用默认 Java 镜像。")
    spec = LangSpec("java", "Main.java", None, shell="javac Main.java && java Main")
