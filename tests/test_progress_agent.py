import asyncio

from harness.events import Progress
from harness.persistence.serialize import event_to_dict
from harness.progress import (
    emit, set_emitter, reset_emitter, set_current_agent, reset_current_agent)


def _capture():
    seen: list = []
    tok = set_emitter(seen.append)
    return seen, tok


def test_sandbox_progress_stamped_with_current_agent():
    seen, tok = _capture()
    at = set_current_agent("研究员")
    try:
        emit(Progress("sandbox", "执行 python", status="running"))
        emit(Progress("subagent:研究员", "调用工具"))  # 非 sandbox 不打标
    finally:
        reset_current_agent(at)
        reset_emitter(tok)
    assert seen[0].agent == "研究员"
    assert seen[1].agent is None


def test_no_stamp_when_agent_unset():
    seen, tok = _capture()
    try:
        emit(Progress("sandbox", "执行"))
    finally:
        reset_emitter(tok)
    assert seen[0].agent is None


def test_does_not_override_existing_agent():
    seen, tok = _capture()
    at = set_current_agent("A")
    try:
        emit(Progress("sandbox", "执行", agent="B"))
    finally:
        reset_current_agent(at)
        reset_emitter(tok)
    assert seen[0].agent == "B"


def test_stamp_propagates_into_async_generator():
    """dispatch 的关键路径：contextvar 要能穿透 async-for 驱动的子 loop 生成器，
    让深层沙箱 emit 读到当前 agent。"""
    async def fake_subrun():
        emit(Progress("sandbox", "执行 run_shell", status="running"))  # 深层工具在迭代中发射
        yield "step"

    async def scenario():
        seen, tok = _capture()
        at = set_current_agent("测试员")
        try:
            async for _ in fake_subrun():
                pass
        finally:
            reset_current_agent(at)
            reset_emitter(tok)
        return seen

    seen = asyncio.run(scenario())
    assert seen and seen[0].agent == "测试员"


def test_serialize_includes_agent():
    d = event_to_dict(Progress("sandbox", "执行", agent="研究员"))
    assert d["type"] == "Progress"
    assert d["data"]["agent"] == "研究员"
