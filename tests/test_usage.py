from harness.usage import (
    Usage, estimate_usage, cost_usd, count_message_tokens, tiered_cost,
    effective_cost, set_price_tiers, reset_price_tiers)
from harness.types import Message, Role, ToolCall


def test_usage_add():
    a = Usage(1, 2, 3)
    b = Usage(10, 20, 30)
    c = a + b
    assert (c.prompt_tokens, c.completion_tokens, c.total_tokens) == (11, 22, 33)


def test_estimate_usage_nonzero():
    u = estimate_usage([Message(role=Role.USER, content="hello world")], "hi there", "gpt-4o-mini")
    assert u.prompt_tokens > 0
    assert u.completion_tokens > 0
    assert u.total_tokens == u.prompt_tokens + u.completion_tokens


def test_cost_usd_with_price():
    u = Usage(1000, 1000, 2000)
    assert cost_usd(u, "m", {"m": [1.0, 2.0]}) == 3.0


def test_cost_usd_without_price_is_none():
    assert cost_usd(Usage(1, 1, 2), "m", {}) is None


# 分层计费（qwen-plus 档位，¥/百万 token）
_TIERS = [[256000, 1.6, 6.4], [1000000, 4.8, 19.2]]


def test_tiered_cost_empty_is_none():
    assert tiered_cost(1000, 1000, []) is None


def test_tiered_cost_first_tier():
    # 输入 100 万分之 1_000_000 → 100万；这里用 1M token 输入 + 0 输出，落第 2 档
    # 先测第 1 档：输入 250K ≤ 256K → 1.6，输出 6.4
    got = tiered_cost(250_000, 100_000, _TIERS)
    assert got == 250_000 / 1_000_000 * 1.6 + 100_000 / 1_000_000 * 6.4


def test_tiered_cost_second_tier_by_input_size():
    # 输入 300K 超过 256K → 落第 2 档：输入 4.8、输出 19.2（输出费率也随档变）
    got = tiered_cost(300_000, 100_000, _TIERS)
    assert got == 300_000 / 1_000_000 * 4.8 + 100_000 / 1_000_000 * 19.2


def test_tiered_cost_above_cap_uses_last_tier():
    # 输入 200 万超过末档上限 → 用末档封顶价
    got = tiered_cost(2_000_000, 0, _TIERS)
    assert got == 2_000_000 / 1_000_000 * 4.8


def test_effective_cost_prefers_price_map():
    # price_map 配了该模型 → 用扁平计费，不看 tiers
    tok = set_price_tiers(_TIERS)
    try:
        assert effective_cost(Usage(1000, 1000, 2000), "m", {"m": [1.0, 2.0]}) == 3.0
    finally:
        reset_price_tiers(tok)


def test_effective_cost_falls_back_to_context_tiers():
    # price_map 空（默认）→ 回退上下文里的分层计费，成本不再恒为 None/0
    tok = set_price_tiers(_TIERS)
    try:
        got = effective_cost(Usage(250_000, 100_000, 350_000), "m", {})
        assert got == tiered_cost(250_000, 100_000, _TIERS)
        assert got > 0
    finally:
        reset_price_tiers(tok)


def test_effective_cost_none_when_no_price_and_no_tiers():
    # price_map 空且未设 tiers → None（保持旧语义，透明）
    assert effective_cost(Usage(1, 1, 2), "m", {}) is None


_M = "gpt-4o-mini"


def test_count_message_tokens_positive_and_monotonic():
    one = count_message_tokens([Message(role=Role.USER, content="讲讲二叉树")], _M)
    two = count_message_tokens(
        [Message(role=Role.USER, content="讲讲二叉树"),
         Message(role=Role.ASSISTANT, content="二叉树是每个节点最多两个子节点的树结构。")], _M)
    assert one > 0
    assert two > one  # 更多消息 → 更多 token


def test_count_message_tokens_includes_tool_calls():
    """关键回归：tool_calls 必须计入，estimate_usage 旧实现漏了它。"""
    plain = Message(role=Role.ASSISTANT, content="好的")
    with_tc = Message(
        role=Role.ASSISTANT, content="好的",
        tool_calls=[ToolCall(id="c1", name="run_python",
                             arguments={"code": "print(sum(range(100)))"})])
    assert count_message_tokens([with_tc], _M) > count_message_tokens([plain], _M)


def test_count_message_tokens_tool_result_message():
    msg = Message(role=Role.TOOL, content="4950", tool_call_id="c1")
    assert count_message_tokens([msg], _M) > 0


def test_count_message_tokens_image_part_not_encoded_as_text():
    """多模态图片按固定近似计，不能把 base64 当文本编码（否则爆表）。"""
    huge_data_url = "data:image/png;base64," + "A" * 100_000
    img_msg = Message(role=Role.USER, content=[
        {"type": "text", "text": "看这张图"},
        {"type": "image_url", "image_url": {"url": huge_data_url}},
    ])
    n = count_message_tokens([img_msg], _M)
    # 图片按固定近似（数百 token），远小于把 10 万字符 base64 当文本编码的量
    assert 100 < n < 5000


def test_count_message_tokens_empty():
    assert count_message_tokens([], _M) >= 0
