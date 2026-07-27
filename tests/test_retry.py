import pytest

from harness.reliability.retry import RetryingModelClient
from harness.llm.base import StreamChunk


class Transient(Exception):
    pass


def _sleeps():
    calls = []

    async def fake_sleep(d):
        calls.append(d)

    return calls, fake_sleep


async def test_retries_then_succeeds(flaky_client, text_turn):
    slept, fake_sleep = _sleeps()
    inner = flaky_client(Transient("timeout"), text_turn("ok"), fail_times=2)
    client = RetryingModelClient(inner, max_retries=2, base_delay=0.1,
                                 sleep=fake_sleep, transient=(Transient,))
    chunks = [c async for c in client.stream([], [])]
    assert inner.calls == 3                       # 1 正常 + 2 重试
    assert "".join(c.text for c in chunks if c.type == "text") == "ok"
    assert len(slept) == 2                        # 退避 2 次
    done = [c for c in chunks if c.type == "done"][0]
    assert done.attempts == 3


async def test_exhausts_and_raises(flaky_client, text_turn):
    slept, fake_sleep = _sleeps()
    inner = flaky_client(Transient("timeout"), text_turn("never"), fail_times=99)
    client = RetryingModelClient(inner, max_retries=2, base_delay=0.1,
                                 sleep=fake_sleep, transient=(Transient,))
    with pytest.raises(Transient):
        [c async for c in client.stream([], [])]
    assert inner.calls == 3                       # 1 + 2 后放弃


async def test_mid_stream_break_not_retried(flaky_client, text_turn):
    slept, fake_sleep = _sleeps()
    inner = flaky_client(Transient("mid"), text_turn("ok"), fail_times=1, mid_stream=True)
    client = RetryingModelClient(inner, max_retries=3, base_delay=0.1,
                                 sleep=fake_sleep, transient=(Transient,))
    with pytest.raises(Transient):
        [c async for c in client.stream([], [])]
    assert inner.calls == 1                        # 已产出 chunk，不重试
    assert slept == []
