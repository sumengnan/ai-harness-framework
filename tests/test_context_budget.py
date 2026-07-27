"""ContextBudget：上下文各层的 token 额度切分。

（与 tests/test_budget.py 无关——那个测的是 reliability 的花费预算，同名不同物。）
"""
from harness.context.budget import ContextBudget


def _b(window=128000, reserve=4096, ratio=0.5, max_prompt=0):
    return ContextBudget(context_window=window, response_reserve=reserve,
                         working_ratio=ratio, max_prompt_tokens=max_prompt)


# ---- 物理约束：窗口 - 回复预留 - system ----

def test_available_subtracts_reserve_and_system():
    assert _b(window=1000, reserve=100).available(50) == 850


def test_working_tokens_is_ratio_of_available():
    assert _b(window=1000, reserve=100, ratio=0.5).working_tokens(50) == 425


def test_never_negative_when_system_prompt_exceeds_window():
    # system 比窗口还大（配置写错时）→ 给 0，不返回负数把下游算崩
    assert _b(window=1000, reserve=100).available(5000) == 0
    assert _b(window=1000, reserve=100).working_tokens(5000) == 0


# ---- 策略约束：max_prompt_tokens（如分档计价的档位阈值）----

def test_max_prompt_tokens_defaults_off():
    """默认 0 = 不设 —— 与该字段引入前的行为逐字节一致，存量部署不受影响。"""
    assert _b().available(1000) == 128000 - 4096 - 1000


def test_max_prompt_caps_available_below_physical_window():
    # 窗口 100 万塞得下，但只想让输入停在 25 万（超档要贵 3 倍）
    b = _b(window=1000000, reserve=64000, max_prompt=250000)
    assert b.available(1000) == 249000          # 250000 - system，而非物理的 935000


def test_max_prompt_also_subtracts_system():
    """它管的是「整个输入」，故须扣掉 system 才可比 —— 否则 system 会白吃掉额度、超出上限。"""
    b = _b(window=1000000, reserve=0, max_prompt=10000)
    assert b.available(3000) == 7000            # 不是 10000


def test_physical_window_still_wins_when_tighter():
    """两个上限取小：策略上限放得再宽，也不能突破物理窗口。"""
    b = _b(window=8000, reserve=1000, max_prompt=999999)
    assert b.available(0) == 7000


def test_max_prompt_larger_than_system_only():
    # 策略上限比 system 还小（配置写错）→ 0，不返回负数
    assert _b(window=1000000, reserve=0, max_prompt=100).available(500) == 0


def test_working_ratio_applies_after_the_cap():
    """比例是在「取小之后」的额度上切，不是先切再取小 —— 否则上限会被比例放大回去。"""
    b = _b(window=1000000, reserve=64000, ratio=0.9, max_prompt=250000)
    assert b.available(1000) == 249000
    assert b.working_tokens(1000) == 224100     # 249000 × 0.9
