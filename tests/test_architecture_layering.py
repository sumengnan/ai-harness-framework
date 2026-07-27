# tests/test_architecture_layering.py
"""分层约束：harness 是可分发的运行时内核，不得依赖上层应用。

pyproject 里只有 `packages = ["src/harness"]` 被打包——发布出去的是 harness，`app/`
只是它的第一个消费者（学习助手）。这条边界目前是干净的（app→harness 120 处引用，
反向 0 处），但它是靠人自觉维持的：随手写一句 `from app.config import AppConfig`
就能把内核焊死在这个产品上，而且不会有任何报错。故在此钉死。
"""
from __future__ import annotations

import ast
import pathlib

import pytest

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "harness"


def _harness_files() -> list[pathlib.Path]:
    return sorted(p for p in _SRC.rglob("*.py") if "__pycache__" not in p.parts)


def test_harness_package_is_present():
    """防呆：路径写错会让下面的参数化静默零用例，看着全绿其实什么都没测。"""
    assert _harness_files(), f"未找到任何 harness 源文件（查找路径：{_SRC}）"


@pytest.mark.parametrize("path", _harness_files(),
                         ids=lambda p: str(p.relative_to(_SRC)))
def test_harness_does_not_import_app(path: pathlib.Path):
    """内核不得 import 上层 app。

    用 AST 而非字符串匹配：注释、文档字符串里提到 app 是正常的（架构说明就会提），
    只有真正的 import 语句才算违规。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bad += [a.name for a in node.names if a.name == "app" or a.name.startswith("app.")]
        elif isinstance(node, ast.ImportFrom):
            # level>0 是包内相对导入（如 from .base import Tool），与 app 无关
            if node.level == 0 and node.module and (
                    node.module == "app" or node.module.startswith("app.")):
                bad.append(node.module)
    assert not bad, (
        f"{path.relative_to(_SRC)} 依赖了上层应用：{bad}。"
        "harness 是可分发内核，业务相关的东西应放在 app/ 侧，"
        "需要内核感知时用鸭子类型/回调注入，不要反向 import。")
