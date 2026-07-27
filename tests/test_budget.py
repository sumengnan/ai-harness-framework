import pytest

from harness.reliability.budget import BudgetTracker, BudgetExceeded
from harness.usage import Usage


def test_no_limits_never_raises():
    b = BudgetTracker()
    b.start()
    b.add_usage(Usage(0, 0, 10**9))
    b.check()  # 不抛


def test_token_budget_exceeded():
    b = BudgetTracker(max_tokens=100)
    b.start()
    b.add_usage(Usage(60, 60, 120))
    with pytest.raises(BudgetExceeded):
        b.check()


def test_time_budget_exceeded_with_fake_clock():
    ticks = iter([0.0, 5.0])  # start, check
    b = BudgetTracker(max_wall_seconds=3.0, clock=lambda: next(ticks))
    b.start()
    with pytest.raises(BudgetExceeded):
        b.check()


def test_total_tokens_accumulates():
    b = BudgetTracker()
    b.start()
    b.add_usage(Usage(1, 1, 2))
    b.add_usage(Usage(3, 3, 6))
    assert b.total_tokens == 8


def test_start_is_idempotent():
    # 共享 budget 场景：主+子多次 start()，墙钟基准只在首次设置。
    # 用可变时钟：首次 start 时=0.0，第二次 start 时已推进到 5.0，check 时=12.0。
    now = [0.0]
    b = BudgetTracker(max_wall_seconds=10.0, clock=lambda: now[0])
    b.start()          # _start=0.0
    now[0] = 5.0
    b.start()          # 幂等：_start 仍=0.0（不被重置到 5.0）
    now[0] = 12.0
    with pytest.raises(BudgetExceeded):
        b.check()      # elapsed=12-0=12>10 → 抛（改前第二次 start 挪基准到 5，7<10 不抛）
