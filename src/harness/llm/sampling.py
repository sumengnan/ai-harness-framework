# src/harness/llm/sampling.py
"""采样参数（temperature / top_p）的请求级覆盖与运行期增量。

与 llm_extra_body 的覆盖机制同形（contextvar，不把参数穿过整条 AgentLoop），但采样参数是
chat.completions 的**顶层**参数、不能塞进 extra_body（会和顶层键撞车），故单开一处。

两层语义：

- **基准**（`sampling_override`）：本轮/本调用该用什么温度。按角色（judge/摘要/判分…）或
  按用户意图（事实问答/创作…）设一次。未设则回退 HarnessConfig.temperature。
- **增量**（`temperature_delta`）：运行期按失败情形临时加减。可嵌套叠加——打转纠偏 +0.2、
  重答第二次 +0.15、planner 重试 -0.1。与基准分离是为了让重试逻辑不必知道基准是多少。

最终值 = clamp(基准 + Σ增量, 0, 1)。

**上限取 1 而非协议上的 2**：OpenAI 协议允许 0~2，但 Anthropic 只到 1，百炼/Qwen 不接受
端点值 2；且 >1 的区间实测只会让输出退化，没有正经用途。统一收在 [0,1] 既跨厂商安全，
又堵住「增量把温度累加到 1.8 变成乱码」这种事故。

top_p 与 temperature **二选一**（两家官方文档都写了 "alter this or top_p but not both"）：
显式设了 top_p 就只发 top_p、不发 temperature，见 resolve_sampling。
"""
from __future__ import annotations

import contextlib
from contextvars import ContextVar

# 温度上下界。上限 1 是刻意的，见模块头。
TEMP_MIN = 0.0
TEMP_MAX = 1.0
TOP_P_MIN = 0.0
TOP_P_MAX = 1.0

# 本轮采样基准：{"temperature": float|None, "top_p": float|None}。空 dict = 不覆盖。
_sampling: ContextVar[dict] = ContextVar("llm_sampling_override", default={})
# 运行期累计增量（只作用于 temperature）。
_temp_delta: ContextVar[float] = ContextVar("llm_temperature_delta", default=0.0)


def clamp_temperature(v: float) -> float:
    """夹到 [0, 1]。所有对外暴露的温度值都必须过这里，包括配置读入与增量累加后。"""
    return min(TEMP_MAX, max(TEMP_MIN, float(v)))


def clamp_top_p(v: float) -> float:
    return min(TOP_P_MAX, max(TOP_P_MIN, float(v)))


def set_sampling_override(temperature: float | None = None,
                          top_p: float | None = None) -> object:
    """设置本轮采样基准，返回 token（用 reset_sampling_override 还原）。

    只覆盖显式传入的那个；两个都传时 top_p 优先（temperature 不发），因为同时发两者是
    厂商明确不建议的用法。传 None = 该项不覆盖，回退 config。"""
    cur = dict(_sampling.get())
    if temperature is not None:
        cur["temperature"] = clamp_temperature(temperature)
    if top_p is not None:
        cur["top_p"] = clamp_top_p(top_p)
    return _sampling.set(cur)


def reset_sampling_override(token) -> None:
    _sampling.reset(token)


def get_sampling_override() -> dict:
    return dict(_sampling.get())


@contextlib.contextmanager
def sampling(temperature: float | None = None, top_p: float | None = None):
    """`with sampling(temperature=0.0): ...` —— 设基准、退出还原。"""
    token = set_sampling_override(temperature=temperature, top_p=top_p)
    try:
        yield
    finally:
        reset_sampling_override(token)


def push_temperature_delta(delta: float) -> object:
    """非 with 形式的增量（供生成器内部使用，如 AgentLoop 检测到打转后升温跑完剩余步骤）。
    必须在同一执行流里 pop_temperature_delta 还原——生成器不自带上下文隔离，不还原会漏给调用方。
    """
    return _temp_delta.set(_temp_delta.get() + float(delta))


def pop_temperature_delta(token) -> None:
    _temp_delta.reset(token)


# 本轮「打转纠偏该升多少温」。由 app 层按配置在请求入口设一次，AgentLoop（含其派生的
# 执行子步、子代理）读取——避免把一个策略常量层层穿过所有 loop 构造点。0=不升温。
_nudge_delta: ContextVar[float] = ContextVar("llm_loop_nudge_delta", default=0.0)


def set_nudge_delta(v: float) -> object:
    return _nudge_delta.set(float(v))


def reset_nudge_delta(token) -> None:
    _nudge_delta.reset(token)


def get_nudge_delta() -> float:
    return _nudge_delta.get()


@contextlib.contextmanager
def temperature_delta(delta: float):
    """`with temperature_delta(+0.2): ...` —— 在当前基准上临时加减，可嵌套叠加。

    升温用于「模型陷在同一条路上」（打转纠偏、整轮重答、单步重试）；降温用于「输出结构
    崩了」（planner 出非法 DAG、JSON 解析失败）。两类失败成因相反，故方向也相反。
    """
    token = _temp_delta.set(_temp_delta.get() + float(delta))
    try:
        yield
    finally:
        _temp_delta.reset(token)


def get_temperature_delta() -> float:
    return _temp_delta.get()


def resolve_sampling(config_temperature: float) -> dict:
    """算出本次请求真正要发的采样参数。返回可直接并进 kwargs 的 dict。

    - 显式设了 top_p → 只发 top_p（temperature 让位，避免两者同发）。
    - 否则发 temperature = clamp(基准 + Σ增量)，基准缺省用 config 值。
    """
    over = _sampling.get()
    if "top_p" in over:
        return {"top_p": clamp_top_p(over["top_p"])}
    base = over.get("temperature", config_temperature)
    return {"temperature": clamp_temperature(base + _temp_delta.get())}


def supports_temperature(model: str, unsupported: list | tuple = ()) -> bool:
    """该模型认不认采样参数（子串匹配模型名）。

    命中的模型一律不发 temperature/top_p——部分推理模型（o1 系、思考模式下的一些 Qwen）
    会因「不支持的参数」直接报错。与 thinking_unsupported_models 同样的兜底思路。
    """
    m = (model or "").lower()
    return not any(x and str(x).lower() in m for x in (unsupported or ()))
