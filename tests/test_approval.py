import asyncio

import pytest

from harness import approval, progress
from harness.events import ApprovalRequired, ApprovalResolved


async def test_no_context_passes_through():
    # 无审批上下文（CLI/测试）→ 直接放行
    assert await approval.request_approval("run_shell", "rm -rf /", "危险") is True


async def test_approve_flow():
    got = []
    etoken = progress.set_emitter(got.append)
    ctoken = approval.set_context(run_id="r1", timeout=5)
    try:
        task = asyncio.create_task(
            approval.request_approval("run_shell", "rm -rf /", "递归删除"))
        await asyncio.sleep(0)  # 让 task 跑到 await，注册 pending 并 emit 事件
        req = next(e for e in got if isinstance(e, ApprovalRequired))
        assert req.run_id == "r1" and req.command == "rm -rf /"
        assert approval.resolve(req.approval_id, True) is True
        assert await task is True
    finally:
        approval.reset_context(ctoken)
        progress.reset_emitter(etoken)
    assert any(isinstance(e, ApprovalResolved) and e.approved for e in got)


async def test_deny_flow():
    got = []
    etoken = progress.set_emitter(got.append)
    ctoken = approval.set_context(run_id="r1", timeout=5)
    try:
        task = asyncio.create_task(
            approval.request_approval("run_shell", "rm -rf /", "递归删除"))
        await asyncio.sleep(0)
        req = next(e for e in got if isinstance(e, ApprovalRequired))
        assert approval.resolve(req.approval_id, False) is True
        assert await task is False
    finally:
        approval.reset_context(ctoken)
        progress.reset_emitter(etoken)


async def test_unknown_id_returns_false():
    assert approval.resolve("does-not-exist", True) is False


async def test_timeout_denies():
    etoken = progress.set_emitter(lambda e: None)
    ctoken = approval.set_context(run_id="r1", timeout=0.05)
    try:
        assert await approval.request_approval("run_shell", "rm -rf /", "危险") is False
    finally:
        approval.reset_context(ctoken)
        progress.reset_emitter(etoken)


async def test_denied_command_is_not_asked_twice():
    """拒绝是人做出的决定，同一条命令不该再问第二次。

    回归：编排器的单步重试会重跑整个子步，工具随之再次 request_approval——用户刚拒绝完，
    同一个弹窗立刻又怼上来，连续几次。"""
    import harness.approval as ap
    from harness.events import ApprovalRequired
    from harness import progress

    got = []
    etoken = progress.set_emitter(got.append)
    ctoken = ap.set_context(run_id="r1", timeout=0.05)   # 无人应答 → 超时拒绝
    try:
        first = await ap.request_approval("run_shell", "rm -rf /workspace", "危险")
        second = await ap.request_approval("run_shell", "rm -rf /workspace", "危险")
        assert first is False and second is False
        asks = [e for e in got if isinstance(e, ApprovalRequired)]
        assert len(asks) == 1, "同一条命令只该弹一次"
    finally:
        ap.reset_context(ctoken)
        progress.reset_emitter(etoken)


async def test_denial_is_scoped_per_run():
    """拒绝只在本轮有效：换一轮（新上下文）用户仍有机会重新决定。"""
    import harness.approval as ap
    from harness.events import ApprovalRequired
    from harness import progress

    got = []
    etoken = progress.set_emitter(got.append)
    try:
        t1 = ap.set_context(run_id="r1", timeout=0.05)
        await ap.request_approval("run_shell", "rm -rf /x", "危险")
        ap.reset_context(t1)
        t2 = ap.set_context(run_id="r2", timeout=0.05)
        await ap.request_approval("run_shell", "rm -rf /x", "危险")
        ap.reset_context(t2)
        assert len([e for e in got if isinstance(e, ApprovalRequired)]) == 2
    finally:
        progress.reset_emitter(etoken)
