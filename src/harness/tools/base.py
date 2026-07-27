from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, ValidationError

from ..types import ToolCall, ToolOutput, ToolResult


class ToolError(Exception):
    """工具主动把本次结果标记为失败：异常内容原样作为结果回传（不加“工具执行出错”前缀）。

    meta：仅供前端展示的元信息（如容器工具失败时也带上用的镜像名），不进模型上下文。
    """

    def __init__(self, message: str = "", *, meta: dict | None = None) -> None:
        super().__init__(message)
        self.meta = meta


class Tool(ABC):
    name: str
    description: str
    Params: type[BaseModel]

    @abstractmethod
    async def run(self, params: BaseModel) -> "str | ToolOutput":
        ...

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.Params.model_json_schema(),
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def tools(self) -> list[Tool]:
        return list(self._tools.values())

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, max_chars: int | None = None) -> None:
        self._registry = registry
        self._max_chars = max_chars

    def _truncate(self, text: str) -> str:
        if self._max_chars is not None and len(text) > self._max_chars:
            return text[: self._max_chars] + "…(已截断)"
        return text

    async def execute(self, call: ToolCall) -> ToolResult:
        tool = self._registry.get(call.name)
        if tool is None:
            return ToolResult(call.id, f"未知工具: {call.name}", is_error=True)
        try:
            params = tool.Params.model_validate(call.arguments)
        except ValidationError as e:
            return ToolResult(call.id, f"参数校验失败: {e}", is_error=True)
        try:
            raw = await tool.run(params)
            if isinstance(raw, ToolOutput):
                text, follow_up, marker, meta = raw.text, raw.follow_up, raw.marker, raw.meta
            else:
                text, follow_up, marker, meta = raw, [], "", None
            # marker 接在截断之后：否则长结果会把机读标记连同正文一起截掉
            text = self._truncate(text)
            return ToolResult(call.id, text + marker, is_error=False,
                              follow_up=follow_up,
                              model_content=text if marker else None, meta=meta)
        except ToolError as e:  # 工具主动标记失败：内容原样回传（失败也带上 meta，如镜像名）
            return ToolResult(call.id, self._truncate(str(e)), is_error=True,
                              meta=getattr(e, "meta", None))
        except Exception as e:  # 工具内部异常兜成 is_error，喂回模型自纠正
            return ToolResult(call.id, f"工具执行出错: {e}", is_error=True)
