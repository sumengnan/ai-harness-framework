from harness.context.clamp import ClampedContextManager
from harness.types import Message, Role
from harness.usage import count_message_tokens

_MODEL = "gpt-4"


class _Inner:
    """假的内层 ContextManager：build() 恒返回预置消息列表（忽略 state）。"""
    def __init__(self, msgs):
        self._msgs = msgs
    def build(self, state):
        return list(self._msgs)


def _m(role, text):
    return Message(role=role, content=text)


def test_clamp_off_is_passthrough():
    msgs = [_m(Role.SYSTEM, "s"), _m(Role.USER, "u1"), _m(Role.USER, "u2")]
    out = ClampedContextManager(_Inner(msgs), _MODEL, 0).build(None)
    assert [x.content for x in out] == ["s", "u1", "u2"]   # <=0 不裁


def test_clamp_under_cap_is_passthrough():
    msgs = [_m(Role.SYSTEM, "sys"), _m(Role.USER, "很短")]
    out = ClampedContextManager(_Inner(msgs), _MODEL, 100000).build(None)
    assert len(out) == 2   # 没超上限，原样返回


def test_clamp_drops_oldest_middle_messages():
    big = "词" * 4000
    msgs = [_m(Role.SYSTEM, "sys"), _m(Role.USER, big), _m(Role.USER, big), _m(Role.USER, "最后一句")]
    cap = count_message_tokens([msgs[0], msgs[-1]], _MODEL) + 20   # 只够 system + 末条
    out = ClampedContextManager(_Inner(msgs), _MODEL, cap).build(None)
    assert out[0].content == "sys" and out[-1].content == "最后一句"   # 保 system 与末条
    assert len(out) < len(msgs)                                       # 丢了中间最旧的
    assert count_message_tokens(out, _MODEL) <= cap                   # 确定性不超上限


def test_clamp_truncates_huge_single_user_message():
    big = "数据" * 8000
    msgs = [_m(Role.SYSTEM, "sys"), _m(Role.USER, big)]
    cap = count_message_tokens([msgs[0]], _MODEL) + 200   # 远小于 big
    out = ClampedContextManager(_Inner(msgs), _MODEL, cap).build(None)
    assert len(out) == 2                                  # 单轮无法丢 user
    assert "省略" in out[-1].content                      # 中段截断标记
    assert out[-1].content.startswith("数据")             # 保头
    assert len(out[-1].content) < len(big)                # 确实变短
    assert count_message_tokens(out, _MODEL) <= cap       # 确定性不超上限
