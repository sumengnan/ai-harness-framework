import pytest
from harness.browser.fake import FakeBrowser
from harness.browser.base import PageResult


async def test_fake_returns_preset():
    fb = FakeBrowser({"http://x/": ("标题", "<p>hi</p>")})
    await fb.start()
    r = await fb.fetch("http://x/", timeout=5, wait_until="load")
    assert isinstance(r, PageResult)
    assert r.title == "标题" and r.html == "<p>hi</p>" and r.final_url == "http://x/"
    await fb.close()


async def test_fake_unknown_url_raises():
    fb = FakeBrowser({})
    with pytest.raises(RuntimeError):
        await fb.fetch("http://missing/", timeout=5, wait_until="load")
