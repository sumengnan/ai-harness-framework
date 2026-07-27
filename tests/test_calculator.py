import pytest

from harness.tools.builtins.calculator import CalculatorTool, safe_eval


def test_safe_eval_basic():
    assert safe_eval("(12+8)*3") == 60
    assert safe_eval("2**3") == 8
    assert safe_eval("-5 + 2") == -3


def test_safe_eval_rejects_code():
    with pytest.raises(ValueError):
        safe_eval("__import__('os').system('ls')")
    with pytest.raises(ValueError):
        safe_eval("open('x')")
    with pytest.raises(ValueError):
        safe_eval("True")


async def test_calculator_tool_run():
    tool = CalculatorTool()
    result = await tool.run(tool.Params(expression="(12+8)*3"))
    assert result == "60"


def test_description_says_numeric_only():
    # 描述必须点明「只做数值」，否则模型会拿它算日期（如 "2026-07-19 + 2"）
    d = CalculatorTool().description
    assert "纯数" in d or "数值" in d
    assert "日期" in d


async def test_calculator_tool_rejects_date_expression_with_clear_message():
    from harness.tools.base import ToolError
    tool = CalculatorTool()
    with pytest.raises(ToolError) as ei:
        await tool.run(tool.Params(expression="2026-07-19 + 2"))
    msg = str(ei.value)
    assert "只做纯数字" in msg and "日期" in msg   # 报错点明原因，便于模型自纠


def test_pow_magnitude_guarded():
    import pytest
    from harness.tools.builtins.calculator import safe_eval
    with pytest.raises(ValueError):
        safe_eval("9 ** 99999999")
    # 正常小幂不受影响
    assert safe_eval("2 ** 10") == 1024
