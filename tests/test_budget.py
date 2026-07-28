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


def test_restore_seeds_tokens_and_wall():
    """resume 场景：装回上次已消耗的量，接着算而不是从零重发一份预算。"""
    now = [0.0]
    b = BudgetTracker(max_tokens=100, max_wall_seconds=10.0, clock=lambda: now[0])
    b.restore(total_tokens=90, wall_seconds=8.0)   # 上次已用 90 tokens / 8 秒
    b.start()
    assert b.total_tokens == 90
    b.check()                                      # 90 未超 100，8s 未超 10s
    b.add_usage(Usage(5, 6, 11))
    with pytest.raises(BudgetExceeded):
        b.check()                                  # 90+11=101 > 100


def test_restore_wall_excludes_downtime():
    """停机时间不计入：只累计真正在跑的墙钟。"""
    now = [1000.0]                                 # 崩溃后过了很久才重启
    b = BudgetTracker(max_wall_seconds=10.0, clock=lambda: now[0])
    b.restore(wall_seconds=8.0)
    b.start()                                      # 基准=1000，停机的 1000 秒不算
    now[0] = 1001.0
    assert b.elapsed_seconds == 9.0                # 8 + 1，不是 8 + 1001
    b.check()
    now[0] = 1003.0
    with pytest.raises(BudgetExceeded):
        b.check()                                  # 8 + 3 = 11 > 10


def test_elapsed_seconds_before_start():
    b = BudgetTracker(clock=lambda: 5.0)
    assert b.elapsed_seconds == 0.0
    b.restore(wall_seconds=3.0)
    assert b.elapsed_seconds == 3.0                # 未 start 时只有装回来的那部分
