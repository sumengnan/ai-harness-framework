import pytest
from pydantic import BaseModel

from harness.tools.base import Tool, ToolRegistry, ToolExecutor
from harness.types import ToolCall


class EchoTool(Tool):
    name = "echo"
    description = "回显文本"

    class Params(BaseModel):
        text: str

    async def run(self, params) -> str:
        return params.text


def test_schema_shape():
    schema = EchoTool().schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "echo"
    assert "text" in schema["function"]["parameters"]["properties"]


def test_registry_register_and_schemas():
    reg = ToolRegistry()
    reg.register(EchoTool())
    assert reg.get("echo") is not None
    assert reg.get("missing") is None
    assert reg.schemas()[0]["function"]["name"] == "echo"


async def test_executor_success():
    reg = ToolRegistry()
    reg.register(EchoTool())
    ex = ToolExecutor(reg)
    result = await ex.execute(ToolCall(id="c1", name="echo", arguments={"text": "hi"}))
    assert result.content == "hi"
    assert result.is_error is False
    assert result.tool_call_id == "c1"


async def test_executor_unknown_tool():
    ex = ToolExecutor(ToolRegistry())
    result = await ex.execute(ToolCall(id="c1", name="nope", arguments={}))
    assert result.is_error is True
    assert "nope" in result.content


async def test_executor_param_validation_error_feeds_back():
    reg = ToolRegistry()
    reg.register(EchoTool())
    ex = ToolExecutor(reg)
    # 缺少必填 text
    result = await ex.execute(ToolCall(id="c1", name="echo", arguments={}))
    assert result.is_error is True
    assert result.tool_call_id == "c1"


async def test_executor_non_dict_arguments_feed_back_is_error():
    # arguments 是合法 JSON 但非 dict（字符串）→ 应优雅降级为 is_error 而非抛异常
    reg = ToolRegistry()
    reg.register(EchoTool())
    ex = ToolExecutor(reg)
    result = await ex.execute(ToolCall(id="c1", name="echo", arguments="not-a-dict"))
    assert result.is_error is True
    assert result.tool_call_id == "c1"


async def test_executor_run_exception_wrapped():
    class BoomTool(Tool):
        name = "boom"
        description = "总是抛错"

        class Params(BaseModel):
            pass

        async def run(self, params) -> str:
            raise RuntimeError("kaboom")

    reg = ToolRegistry()
    reg.register(BoomTool())
    ex = ToolExecutor(reg)
    result = await ex.execute(ToolCall(id="c1", name="boom", arguments={}))
    assert result.is_error is True
    assert "kaboom" in result.content


async def test_tool_result_truncated_when_too_long():
    class LongTool(Tool):
        name = "long"
        description = "返回超长文本"

        class Params(BaseModel):
            pass

        async def run(self, params) -> str:
            return "x" * 100

    reg = ToolRegistry()
    reg.register(LongTool())
    ex = ToolExecutor(reg, max_chars=20)
    result = await ex.execute(ToolCall(id="c1", name="long", arguments={}))
    assert len(result.content) <= 20 + len("…(已截断)")
    assert result.content.endswith("…(已截断)")
    assert result.is_error is False
