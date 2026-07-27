from __future__ import annotations

import ast
import operator

from pydantic import BaseModel

from ..base import Tool, ToolError

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_MAX_POW_EXPONENT = 1000  # 防 9**99999999 类 DoS


def _eval(node: ast.AST):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left = _eval(node.left)
        right = _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POW_EXPONENT:
            raise ValueError(f"幂运算指数过大（|{right}| > {_MAX_POW_EXPONENT}）")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("不支持的表达式（仅允许数字与 + - * / ** % 和括号）")


def safe_eval(expression: str):
    """受限 AST 求值，绝不使用 eval，杜绝任意代码执行。"""
    return _eval(ast.parse(expression, mode="eval").body)


class CalculatorTool(Tool):
    name = "calculator"
    description = (
        "对纯数字做算术计算，支持 + - * / ** % 和括号（如 3*(4+5)、2**10、100%7）。"
        "只接受数值表达式：不支持日期/时间运算（如 “2026-07-19 + 2” 这类会失败）、"
        "变量、函数、单位或文本。日期加减、时间换算等请勿传入本工具，另行推算。")

    class Params(BaseModel):
        expression: str

    async def run(self, params: "CalculatorTool.Params") -> str:
        try:
            return str(safe_eval(params.expression))
        except (SyntaxError, ValueError, TypeError) as e:
            # 非纯数值表达式（如日期 “2026-07-19 + 2”、变量、函数、文本）会在解析/求值时抛错。
            # 回一句清楚的说明而非 cryptic 报错，让模型知道该换算法、别再把它塞进计算器。
            raise ToolError(
                f"无法计算 “{params.expression}”：本工具只做纯数字算术（+ - * / ** % 和括号），"
                f"不支持日期/时间运算、变量、函数或文本。请改用纯数字表达式，或另行推算。") from e
