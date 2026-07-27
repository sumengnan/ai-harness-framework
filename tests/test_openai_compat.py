import pytest

from harness.llm.openai_compat import OpenAICompatibleClient
from harness.config import HarnessConfig
from harness.types import Message, Role


class _FakeDelta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, delta):
        self.delta = delta


class _FakeEvent:
    def __init__(self, delta):
        self.choices = [_FakeChoice(delta)]


class _FakeToolCall:
    def __init__(self, index, id, name, arguments):
        self.index = index
        self.id = id
        self.function = type("F", (), {"name": name, "arguments": arguments})()


async def _fake_stream(events):
    for e in events:
        yield e


async def test_stream_normalizes_text_and_tool_and_done(monkeypatch):
    cfg = HarnessConfig(api_key="k")
    client = OpenAICompatibleClient(cfg)

    events = [
        _FakeEvent(_FakeDelta(content="你好")),
        _FakeEvent(_FakeDelta(tool_calls=[_FakeToolCall(0, "c1", "calculator", '{"e')])),
        _FakeEvent(_FakeDelta(tool_calls=[_FakeToolCall(0, None, None, 'xp": "1+1"}')])),
    ]

    async def fake_create(**kwargs):
        assert kwargs["stream"] is True
        assert kwargs["messages"][0]["role"] == "user"
        return _fake_stream(events)

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)

    out = [c async for c in client.stream([Message(role=Role.USER, content="hi")], [])]
    assert out[0].type == "text" and out[0].text == "你好"
    assert out[1].type == "tool_call" and out[1].tool_call_delta.name == "calculator"
    assert out[2].tool_call_delta.arguments == 'xp": "1+1"}'
    assert out[-1].type == "done"


class _ReasoningDelta:
    def __init__(self, reasoning=None, content=None):
        self.reasoning_content = reasoning
        self.content = content
        self.tool_calls = None


async def test_stream_yields_reasoning_before_text(monkeypatch):
    # 思考模式：reasoning_content 先于 content，单独产出 reasoning chunk（不计入正文）
    cfg = HarnessConfig(api_key="k")
    client = OpenAICompatibleClient(cfg)
    events = [
        _FakeEvent(_ReasoningDelta(reasoning="先想一下")),
        _FakeEvent(_ReasoningDelta(content="最终答案")),
    ]

    async def fake_create(**kwargs):
        return _fake_stream(events)

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    out = [c async for c in client.stream([Message(role=Role.USER, content="hi")], [])]
    assert out[0].type == "reasoning" and out[0].text == "先想一下"
    assert any(c.type == "text" and c.text == "最终答案" for c in out)


class _FakeUsage:
    def __init__(self, p, c, t):
        self.prompt_tokens = p
        self.completion_tokens = c
        self.total_tokens = t


class _FakeEventUsage:
    """带 usage、无 choices 的尾 chunk（include_usage 行为）。"""
    def __init__(self, usage):
        self.choices = []
        self.usage = usage


async def test_done_chunk_uses_real_usage(monkeypatch):
    cfg = HarnessConfig(api_key="k")
    client = OpenAICompatibleClient(cfg)
    events = [
        _FakeEvent(_FakeDelta(content="你好")),
        _FakeEventUsage(_FakeUsage(11, 7, 18)),
    ]

    async def fake_create(**kwargs):
        assert kwargs["stream_options"] == {"include_usage": True}
        return _fake_stream(events)

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    out = [c async for c in client.stream([Message(role=Role.USER, content="hi")], [])]
    done = [c for c in out if c.type == "done"][0]
    assert (done.usage.prompt_tokens, done.usage.completion_tokens, done.usage.total_tokens) == (11, 7, 18)


async def test_done_chunk_falls_back_to_tiktoken(monkeypatch):
    cfg = HarnessConfig(api_key="k")
    client = OpenAICompatibleClient(cfg)
    events = [_FakeEvent(_FakeDelta(content="hello world"))]  # 无 usage 尾 chunk

    async def fake_create(**kwargs):
        return _fake_stream(events)

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    out = [c async for c in client.stream([Message(role=Role.USER, content="hi")], [])]
    done = [c for c in out if c.type == "done"][0]
    assert done.usage is not None
    assert done.usage.total_tokens > 0   # tiktoken 估算


async def test_thinking_adapted_to_deepseek_disabled(monkeypatch):
    # DeepSeek 端点：enable_thinking 意图翻译成 thinking={"type":"disabled"}（DeepSeek 不认前者）
    cfg = HarnessConfig(api_key="k", base_url="https://api.deepseek.com",
                        llm_extra_body={"enable_thinking": False})
    client = OpenAICompatibleClient(cfg)
    events = [_FakeEvent(_FakeDelta(content="hi"))]

    async def fake_create(**kwargs):
        assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
        return _fake_stream(events)

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    [c async for c in client.stream([Message(role=Role.USER, content="hi")], [])]


async def test_thinking_adapted_to_deepseek_enabled(monkeypatch):
    from harness.llm.openai_compat import reset_extra_body_override, set_extra_body_override
    cfg = HarnessConfig(api_key="k", base_url="https://api.deepseek.com/v1")
    client = OpenAICompatibleClient(cfg)
    events = [_FakeEvent(_FakeDelta(content="hi"))]

    async def fake_create(**kwargs):
        assert kwargs["extra_body"] == {"thinking": {"type": "enabled"}}
        return _fake_stream(events)

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    tok = set_extra_body_override({"enable_thinking": True})
    try:
        [c async for c in client.stream([Message(role=Role.USER, content="hi")], [])]
    finally:
        reset_extra_body_override(tok)


_QWEN_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


async def test_thinking_kept_as_enable_thinking_for_qwen(monkeypatch):
    # 百炼(dashscope) 端点原生认 enable_thinking，保持不变、不翻译成 thinking。
    # model 必须显式钉死：_adapt_thinking 模型名优先判厂商，不给就会读进 .env 的
    # HARNESS_MODEL（开发机可能是 deepseek），断言随环境漂移。
    cfg = HarnessConfig(api_key="k", model="qwen-turbo", base_url=_QWEN_URL,
                        llm_extra_body={"enable_thinking": False})
    client = OpenAICompatibleClient(cfg)
    events = [_FakeEvent(_FakeDelta(content="hi"))]

    async def fake_create(**kwargs):
        assert kwargs["extra_body"] == {"enable_thinking": False}
        return _fake_stream(events)

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    [c async for c in client.stream([Message(role=Role.USER, content="hi")], [])]


async def test_extra_body_passed_when_configured(monkeypatch):
    # 厂商钉死为 Qwen：本例验证「非 DeepSeek 端点原样透传」，厂商若由 .env 决定则断言无意义
    cfg = HarnessConfig(api_key="k", model="qwen-turbo", base_url=_QWEN_URL,
                        llm_extra_body={"enable_thinking": False})
    client = OpenAICompatibleClient(cfg)
    events = [_FakeEvent(_FakeDelta(content="hi"))]

    async def fake_create(**kwargs):
        assert kwargs["extra_body"] == {"enable_thinking": False}   # 透传
        return _fake_stream(events)

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    [c async for c in client.stream([Message(role=Role.USER, content="hi")], [])]


async def test_extra_body_override_merges_over_config(monkeypatch):
    from harness.llm.openai_compat import (
        reset_extra_body_override,
        set_extra_body_override,
    )
    cfg = HarnessConfig(api_key="k", model="qwen-turbo", base_url=_QWEN_URL,
                        llm_extra_body={"a": 1})
    client = OpenAICompatibleClient(cfg)
    events = [_FakeEvent(_FakeDelta(content="hi"))]

    async def fake_create(**kwargs):
        assert kwargs["extra_body"] == {"a": 1, "enable_thinking": True}  # 覆盖叠加在全局上
        return _fake_stream(events)

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    tok = set_extra_body_override({"enable_thinking": True})
    try:
        [c async for c in client.stream([Message(role=Role.USER, content="hi")], [])]
    finally:
        reset_extra_body_override(tok)


async def test_extra_body_omitted_by_default(monkeypatch):
    cfg = HarnessConfig(api_key="k")            # 默认空 → 不加 extra_body（零行为变更）
    client = OpenAICompatibleClient(cfg)
    events = [_FakeEvent(_FakeDelta(content="hi"))]

    async def fake_create(**kwargs):
        assert "extra_body" not in kwargs
        return _fake_stream(events)

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    [c async for c in client.stream([Message(role=Role.USER, content="hi")], [])]


async def test_include_usage_disabled_omits_stream_options(monkeypatch):
    cfg = HarnessConfig(api_key="k", include_usage=False)
    client = OpenAICompatibleClient(cfg)
    events = [_FakeEvent(_FakeDelta(content="hello world"))]  # 无 usage 尾 chunk

    async def fake_create(**kwargs):
        assert "stream_options" not in kwargs
        return _fake_stream(events)

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    out = [c async for c in client.stream([Message(role=Role.USER, content="hi")], [])]
    done = [c for c in out if c.type == "done"][0]
    assert done.usage is not None
    assert done.usage.total_tokens > 0   # tiktoken 兜底


# ---- _adapt_thinking：按模型名判厂商 + 不支持思考参数的模型白名单 ----

def test_adapt_thinking_detects_deepseek_by_model_name():
    from harness.llm.openai_compat import _adapt_thinking
    # 统一网关：base_url 判不出厂商，但模型名含 deepseek → 翻成 thinking={type}
    out = _adapt_thinking({"enable_thinking": True}, "https://gateway.example.com/v1",
                          model="deepseek-v4-flash")
    assert out == {"thinking": {"type": "enabled"}}


def test_adapt_thinking_keeps_enable_thinking_for_qwen_on_gateway():
    from harness.llm.openai_compat import _adapt_thinking
    # 同一网关上的 qwen 模型 → 保持 enable_thinking（不被 deepseek 规则误伤）
    out = _adapt_thinking({"enable_thinking": False}, "https://gateway.example.com/v1",
                          model="qwen-turbo")
    assert out == {"enable_thinking": False}


def test_adapt_thinking_strips_param_for_unsupported_model():
    from harness.llm.openai_compat import _adapt_thinking
    # 白名单命中（子串匹配）→ 既不发 enable_thinking 也不发 thinking
    out = _adapt_thinking({"enable_thinking": True}, "https://dashscope.example.com/v1",
                          model="qwen-turbo", unsupported=["qwen-turbo", "-flash"])
    assert out == {}
    # deepseek 模型也一样：命中白名单就不翻译
    out2 = _adapt_thinking({"enable_thinking": False}, "https://api.deepseek.com",
                           model="deepseek-v4-flash", unsupported=["-flash"])
    assert out2 == {}


def test_adapt_thinking_noop_when_no_intent():
    from harness.llm.openai_compat import _adapt_thinking
    assert _adapt_thinking({"a": 1}, "https://api.deepseek.com", model="deepseek-x") == {"a": 1}
