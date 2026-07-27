"""采样参数（temperature/top_p）的机制层：覆盖、增量、边界夹取、豁免。"""
import pytest

from harness.llm.sampling import (
    TEMP_MAX,
    clamp_temperature,
    clamp_top_p,
    get_nudge_delta,
    pop_temperature_delta,
    push_temperature_delta,
    reset_nudge_delta,
    resolve_sampling,
    sampling,
    set_nudge_delta,
    supports_temperature,
    temperature_delta,
)


# ── 边界夹取 ────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("raw,want", [
    (0.7, 0.7), (0, 0.0), (1, 1.0),
    (1.5, 1.0),      # 协议允许到 2，但我们收在 1：>1 只会让输出退化
    (2.0, 1.0),
    (-0.3, 0.0),
])
def test_clamp_temperature(raw, want):
    assert clamp_temperature(raw) == want


@pytest.mark.parametrize("raw,want", [(0.9, 0.9), (1.7, 1.0), (-1, 0.0)])
def test_clamp_top_p(raw, want):
    assert clamp_top_p(raw) == want


def test_no_override_uses_config_value():
    assert resolve_sampling(0.7) == {"temperature": 0.7}


def test_override_replaces_config_value():
    with sampling(temperature=0.0):
        assert resolve_sampling(0.7) == {"temperature": 0.0}
    assert resolve_sampling(0.7) == {"temperature": 0.7}      # 退出即还原


def test_override_is_clamped_on_the_way_in():
    with sampling(temperature=3.0):
        assert resolve_sampling(0.7) == {"temperature": TEMP_MAX}


# ── top_p 与 temperature 二选一 ─────────────────────────────────────────────────
def test_top_p_wins_and_temperature_is_not_sent():
    """两家官方都写了「只调其一」，故设了 top_p 就不发 temperature。"""
    with sampling(top_p=0.9):
        out = resolve_sampling(0.7)
    assert out == {"top_p": 0.9}
    assert "temperature" not in out


def test_top_p_wins_even_when_both_given():
    with sampling(temperature=0.2, top_p=0.5):
        assert resolve_sampling(0.7) == {"top_p": 0.5}


# ── 增量 ───────────────────────────────────────────────────────────────────────
def test_delta_adds_to_base():
    with sampling(temperature=0.2), temperature_delta(0.15):
        assert resolve_sampling(0.7)["temperature"] == pytest.approx(0.35)


def test_deltas_nest_and_accumulate():
    with sampling(temperature=0.2):
        with temperature_delta(0.15):
            with temperature_delta(0.15):
                assert resolve_sampling(0.7)["temperature"] == pytest.approx(0.5)
            assert resolve_sampling(0.7)["temperature"] == pytest.approx(0.35)
        assert resolve_sampling(0.7)["temperature"] == pytest.approx(0.2)


def test_delta_cannot_push_past_one():
    """升温累加撞上限也只到 1——这正是上限收在 1 要挡住的事故。"""
    with sampling(temperature=0.9), temperature_delta(0.5):
        assert resolve_sampling(0.7)["temperature"] == 1.0


def test_negative_delta_cannot_go_below_zero():
    with sampling(temperature=0.1), temperature_delta(-0.3):
        assert resolve_sampling(0.7)["temperature"] == 0.0


def test_delta_applies_to_config_base_without_override():
    with temperature_delta(0.2):
        assert resolve_sampling(0.7)["temperature"] == pytest.approx(0.9)


def test_delta_does_not_apply_to_top_p():
    with sampling(top_p=0.9), temperature_delta(0.5):
        assert resolve_sampling(0.7) == {"top_p": 0.9}


def test_push_pop_delta_roundtrip():
    token = push_temperature_delta(0.2)
    assert resolve_sampling(0.5)["temperature"] == pytest.approx(0.7)
    pop_temperature_delta(token)
    assert resolve_sampling(0.5)["temperature"] == 0.5


# ── 纠偏升温幅度（本轮策略值）──────────────────────────────────────────────────
def test_nudge_delta_defaults_to_zero():
    assert get_nudge_delta() == 0.0


def test_nudge_delta_set_and_reset():
    token = set_nudge_delta(0.2)
    assert get_nudge_delta() == 0.2
    reset_nudge_delta(token)
    assert get_nudge_delta() == 0.0


# ── 豁免名单 ───────────────────────────────────────────────────────────────────
def test_unsupported_model_matches_by_substring():
    assert not supports_temperature("o1-preview", ["o1-"])
    assert supports_temperature("qwen-plus", ["o1-"])


def test_empty_unsupported_list_allows_everything():
    assert supports_temperature("anything", [])
    assert supports_temperature("anything")
