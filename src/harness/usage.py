# src/harness/usage.py
from __future__ import annotations

import contextvars
import json
from dataclasses import dataclass

from .types import Message


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.total_tokens + other.total_tokens,
        )


def _get_encoding(model: str):
    import tiktoken

    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def _encode_len(text: str, model: str) -> int:
    return len(_get_encoding(model).encode(text))


# 按 OpenAI chat 计费口径的固定开销：每条消息的结构包裹 + 回复引导。
_TOKENS_PER_MESSAGE = 3
_TOKENS_REPLY_PRIMING = 3
# 低精度图片的近似 token 下限；避免把 base64 data URL 当文本编码撑爆预算。
_IMAGE_TOKENS = 765


def _count_content_tokens(content, enc) -> int:
    """content 可为 str / None / 多模态 content-parts 列表。"""
    if content is None:
        return 0
    if isinstance(content, str):
        return len(enc.encode(content))
    total = 0
    for part in content:
        if not isinstance(part, dict):
            total += len(enc.encode(str(part)))
            continue
        ptype = part.get("type")
        if ptype == "text":
            total += len(enc.encode(part.get("text", "")))
        elif ptype == "image_url":
            total += _IMAGE_TOKENS  # 固定近似，不编码 base64 本身
        else:
            total += len(enc.encode(json.dumps(part, ensure_ascii=False)))
    return total


def count_message_tokens(messages: list[Message], model: str) -> int:
    """精确估算一组消息作为 prompt 的 token 数（含 tool_calls 与多模态图片）。

    比 estimate_usage 更完整：estimate_usage 只算了 m.content，漏了 tool_calls，
    也会把图片 base64 当文本。上下文预算裁剪需要这个更贴近真实计费的口径。
    """
    if not messages:
        return 0
    enc = _get_encoding(model)
    total = 0
    for m in messages:
        total += _TOKENS_PER_MESSAGE
        total += len(enc.encode(m.role.value))
        total += _count_content_tokens(m.content, enc)
        for tc in m.tool_calls:
            total += len(enc.encode(tc.name))
            total += len(enc.encode(json.dumps(tc.arguments, ensure_ascii=False)))
            total += 4  # 每个 tool_call 的结构包裹近似
        if m.tool_call_id:
            total += len(enc.encode(m.tool_call_id))
    total += _TOKENS_REPLY_PRIMING
    return total


def estimate_usage(messages: list[Message], completion: str, model: str) -> Usage:
    """端点不返回真实 usage 时的 tiktoken 估算（近似值，仅用于预算保护）。"""
    prompt_text = "\n".join(m.content or "" for m in messages)
    p = _encode_len(prompt_text, model)
    c = _encode_len(completion, model)
    return Usage(prompt_tokens=p, completion_tokens=c, total_tokens=p + c)


def cost_usd(usage: Usage, model: str, price_map: dict) -> float | None:
    price = price_map.get(model)
    if not price:
        return None
    in_per_1k, out_per_1k = price
    return usage.prompt_tokens / 1000 * in_per_1k + usage.completion_tokens / 1000 * out_per_1k


# 分层计费 tiers 的进程内旁路（contextvar）：扁平 price_map 未配某模型价时的实时成本回退。
# 由应用层（chat 路由的 pump()）在进入模型循环前 set，主循环与其派生的所有子任务（executor、
# synthesize、planner/critic completer、dispatch 子循环——皆由 create_task 拷贝上下文）都能读到，
# 无需把 tiers 逐个穿过构造函数。未 set 时为 None，effective_cost 退回「无价即 None」的旧语义。
_price_tiers: contextvars.ContextVar = contextvars.ContextVar("price_tiers", default=None)
# 按模型的分层计费表 {model_name: tiers} 与扁平价表 {model_name: [in,out]/1k}：让编排器各子调用
# （executor/simple 走快速模型、synthesize/planner 走主模型…）按各自模型名精确计价，而不再共用一张表。
# 与 _price_tiers 一样由 chat 路由的 pump() set，主循环及其派生子任务（create_task 拷贝上下文）都读得到。
_price_tiers_by_model: contextvars.ContextVar = contextvars.ContextVar("price_tiers_by_model", default=None)
_price_map_ctx: contextvars.ContextVar = contextvars.ContextVar("price_map_ctx", default=None)


def set_price_tiers(tiers):
    """设置实时成本回退用的（默认）分层计费表，返回 token 供 reset。"""
    return _price_tiers.set(tiers)


def reset_price_tiers(token) -> None:
    _price_tiers.reset(token)


def set_pricing(*, price_map=None, tiers=None, tiers_by_model=None):
    """一次性设三份计价上下文（扁平价表 / 默认分层表 / 按模型分层表），返回 tokens 元组供 reset_pricing。"""
    return (_price_map_ctx.set(price_map),
            _price_tiers.set(tiers),
            _price_tiers_by_model.set(tiers_by_model))


def reset_pricing(tokens) -> None:
    m, t, bm = tokens
    _price_map_ctx.reset(m)
    _price_tiers.reset(t)
    _price_tiers_by_model.reset(bm)


def effective_cost(usage: Usage, model: str, price_map: dict) -> float | None:
    """实时成本，**按模型**计价：
    1) 扁平价表（优先入参 price_map，其次上下文里的 _price_map_ctx）按 model 精确取价；
    2) 无扁平价 → 该模型的 per-model 分层表（_price_tiers_by_model[model]），
    3) 再无 → 全局默认分层表（_price_tiers）。都没有则返回 None。

    编排器各子调用不显式传 price_map，靠 pump() 设的上下文按各自 model 名精确计价。
    """
    pm = price_map or _price_map_ctx.get() or {}
    c = cost_usd(usage, model, pm)
    if c is not None:
        return c
    by = _price_tiers_by_model.get() or {}
    tiers = by.get(model) or _price_tiers.get()
    if tiers:
        return tiered_cost(usage.prompt_tokens, usage.completion_tokens, tiers)
    return None


def tiered_cost(prompt_tokens: int, completion_tokens: int, tiers: list) -> float | None:
    """分层计费：按输入长度选档，返回该次调用的估算成本（货币由调用方约定）。

    tiers 为 [[输入上限tokens, 输入价/百万token, 输出价/百万token], ...]，须升序。
    取首个「输入 tokens ≤ 上限」的档；都超过则用末档（封顶价）。tiers 为空返回 None。
    """
    if not tiers:
        return None
    in_rate, out_rate = tiers[-1][1], tiers[-1][2]
    for limit, ir, orr in tiers:
        if prompt_tokens <= limit:
            in_rate, out_rate = ir, orr
            break
    return prompt_tokens / 1_000_000 * in_rate + completion_tokens / 1_000_000 * out_rate
