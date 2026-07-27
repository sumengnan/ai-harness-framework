from __future__ import annotations

import contextlib
import logging
from contextvars import ContextVar
from typing import AsyncIterator

from openai import AsyncOpenAI

from ..config import HarnessConfig
from ..types import Message
from ..usage import Usage, estimate_usage
from .base import StreamChunk, ToolCallDelta
from .sampling import resolve_sampling, supports_temperature

logger = logging.getLogger(__name__)

# 本轮请求级的 extra_body 覆盖（叠加在全局 config.llm_extra_body 之上）。
# 用 contextvar 传递，避免把参数穿过整条 AgentLoop；如聊天页「思考模式」开关按请求控制
# 思考意图 {"enable_thinking": bool}（厂商中立，发送前由 _adapt_thinking 按端点翻译成
# 各家参数）。默认空=不覆盖。
_extra_body_override: ContextVar[dict] = ContextVar("llm_extra_body_override", default={})


def set_extra_body_override(d: dict):
    """设置本轮 extra_body 覆盖，返回 token（用 reset_extra_body_override 还原）。"""
    return _extra_body_override.set(dict(d or {}))


def reset_extra_body_override(token) -> None:
    _extra_body_override.reset(token)


def get_extra_body_override() -> dict:
    return _extra_body_override.get()


@contextlib.contextmanager
def json_output():
    """强制本轮 LLM 调用输出合法 JSON（response_format=json_object），叠加在当前 extra_body
    覆盖上、调用后还原。比「prompt 要求 + 手工去围栏解析」更稳，少踩「模型混入解释文字/
    markdown 围栏导致解析失败」的坑。

    注意 json_object 模式要求返回顶层是 JSON **对象**：需要数组的调用点必须让 prompt 把数组
    包进对象信封（如 {"items":[...]}）再取键，不能直接套。端点不支持 response_format 时，
    由各调用点自身的 try/except 兜底（解析失败即降级，不因基建差异中断）。"""
    token = set_extra_body_override(
        {**get_extra_body_override(), "response_format": {"type": "json_object"}})
    try:
        yield
    finally:
        reset_extra_body_override(token)


def _adapt_thinking(extra_body: dict, base_url: str, model: str = "",
                    unsupported: list | tuple = ()) -> dict:
    """把厂商中立的思考意图 enable_thinking(bool) 翻译成当前端点/模型认识的参数。

    各厂商开关「思考模式」的参数不同：Qwen/百炼(dashscope) 用 enable_thinking，
    DeepSeek 用 thinking={"type": "enabled"/"disabled"}。上游（聊天页开关、judge/
    grounding/fast 各 completer）统一只表达 enable_thinking 意图，落成哪种参数由这里适配：

    - 厂商识别**模型名优先、再看 base_url**：统一网关下一个 base_url 转发多家模型时，base_url
      判不出厂商，靠模型名（如含 "deepseek"）区分；同端点混用多厂商也能翻对。
    - unsupported：不支持思考切换的模型（子串匹配模型名）——直接去掉思考参数，避免端点因
      「未知参数」报错。上游无需改动，新增厂商/例外在此扩展。
    """
    if "enable_thinking" not in extra_body:
        return extra_body
    eb = dict(extra_body)
    m = (model or "").lower()
    if any(x and str(x).lower() in m for x in (unsupported or ())):
        eb.pop("enable_thinking", None)   # 该模型不认思考参数 → 一律不发
        return eb
    want = bool(eb["enable_thinking"])
    host = (base_url or "").lower()
    if "deepseek" in m or "deepseek" in host:   # DeepSeek 口径：模型名优先，再看端点
        eb.pop("enable_thinking", None)
        eb["thinking"] = {"type": "enabled" if want else "disabled"}
    # 否则保持 enable_thinking 原样（Qwen/百炼 原生即认）
    return eb


class OpenAICompatibleClient:
    """基于 openai async SDK 的实现，base_url 可指向任意兼容端点。

    职责：把"消息列表 + 工具 schema"变成归一化的 StreamChunk 流。
    不做重试、不做路由。
    """

    def __init__(self, config: HarnessConfig) -> None:
        self._config = config
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.request_timeout,
        )

    async def stream(
        self, messages: list[Message], tools: list[dict]
    ) -> AsyncIterator[StreamChunk]:
        kwargs: dict = {
            "model": self._config.model,
            "messages": [m.to_openai() for m in messages],
            "stream": True,
        }
        # 采样参数：config.temperature 是基准，可被本轮 override（按角色/意图）与运行期
        # delta（打转纠偏升温、结构解析失败降温）改写，最终恒被夹在 [0,1]，见 llm/sampling.py。
        # 命中豁免名单的模型（o1 系等推理模型）一个都不发，避免端点因未知参数报错。
        if supports_temperature(self._config.model,
                                getattr(self._config, "sampling_unsupported_models", ())):
            kwargs.update(resolve_sampling(self._config.temperature))
        if self._config.include_usage:
            kwargs["stream_options"] = {"include_usage": True}
        if tools:
            kwargs["tools"] = tools
        # 全局 extra_body（config）叠加本轮 override（contextvar，如聊天页的思考模式开关），
        # 再按端点厂商把思考意图翻译成对应参数（enable_thinking / thinking），见 _adapt_thinking
        extra_body = _adapt_thinking(
            {**self._config.llm_extra_body, **get_extra_body_override()},
            self._config.base_url, self._config.model,
            getattr(self._config, "thinking_unsupported_models", ()))
        if extra_body:
            kwargs["extra_body"] = extra_body

        completion_parts: list[str] = []
        captured = None
        finish_reason = None
        produced = False           # 本轮是否产出过任何 text / tool_call
        stream = await self._client.chat.completions.create(**kwargs)
        async for event in stream:
            if getattr(event, "usage", None):
                captured = event.usage
            if not event.choices:
                continue
            choice = event.choices[0]
            if getattr(choice, "finish_reason", None):
                finish_reason = choice.finish_reason
            delta = choice.delta
            # 思考模式（Qwen3/DeepSeek 等）：推理内容走 reasoning_content，先于正文产出。
            # 不计入 completion_parts（不是正文），单独作为 reasoning chunk 让前端展示「思考中」。
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                yield StreamChunk(type="reasoning", text=reasoning)
            if getattr(delta, "content", None):
                produced = True
                completion_parts.append(delta.content)
                yield StreamChunk(type="text", text=delta.content)
            for tc in (getattr(delta, "tool_calls", None) or []):
                produced = True
                fn = getattr(tc, "function", None)
                yield StreamChunk(type="tool_call", tool_call_delta=ToolCallDelta(
                    index=tc.index,
                    id=getattr(tc, "id", None),
                    name=getattr(fn, "name", None) if fn else None,
                    arguments=getattr(fn, "arguments", None) if fn else None,
                ))

        # 空产出留证：上游返回 200 但既无文本也无工具调用（常见于内容安全拦截
        # finish_reason=content_filter、被截断 length，或部分兼容端点把错误塞进 200 流）。
        # 这里记 warning 便于定位「AI 只回…、2 秒完成」这类静默失败的真实上游原因。
        if not produced:
            logger.warning(
                "模型空产出：无 text/tool_call（model=%s, finish_reason=%s）——"
                "可能触发内容安全策略、被截断或上游异常", self._config.model, finish_reason)

        if captured is not None:
            usage = Usage(captured.prompt_tokens, captured.completion_tokens, captured.total_tokens)
        else:
            usage = estimate_usage(messages, "".join(completion_parts), self._config.model)
        yield StreamChunk(type="done", usage=usage)
