import pytest
from harness.sandbox.local import LocalSandbox
from harness.sandbox.base import SandboxError


async def test_exec_and_file_roundtrip():
    sb = LocalSandbox()
    await sb.start()
    try:
        r = await sb.exec(["echo", "hi"], timeout=5)
        assert r.exit_code == 0 and "hi" in r.stdout and r.timed_out is False
        await sb.write_file("a.txt", "hello")
        assert await sb.read_file("a.txt") == "hello"
        assert "a.txt" in await sb.list_files(".")
    finally:
        await sb.close()


async def test_exec_shell_via_sh_c():
    sb = LocalSandbox()
    await sb.start()
    try:
        r = await sb.exec(["sh", "-c", "echo $((1+1))"], timeout=5)
        assert "2" in r.stdout
    finally:
        await sb.close()


async def test_timeout_kills():
    sb = LocalSandbox()
    await sb.start()
    try:
        r = await sb.exec(["sleep", "5"], timeout=0.3)
        assert r.timed_out is True
    finally:
        await sb.close()


async def test_path_escape_rejected():
    sb = LocalSandbox()
    await sb.start()
    try:
        with pytest.raises(SandboxError):
            await sb.read_file("../../etc/passwd")
    finally:
        await sb.close()


async def test_adopted_workspace_is_used_and_not_deleted(tmp_path):
    """接管既有目录：工作区就是传入的路径，close() 后目录仍在。"""
    (tmp_path / "existing.txt").write_text("kept", encoding="utf-8")
    sb = LocalSandbox(workspace=str(tmp_path))
    await sb.start()
    try:
        assert sb.workspace == str(tmp_path)
        assert await sb.read_file("existing.txt") == "kept"
    finally:
        await sb.close()
    assert tmp_path.exists()
    assert (tmp_path / "existing.txt").read_text(encoding="utf-8") == "kept"


async def test_default_workspace_still_temp_and_deleted():
    """不传 workspace 时行为完全不变：临时目录 + close() 删除。"""
    sb = LocalSandbox()
    await sb.start()
    ws = sb.workspace
    assert ws and ws != ""
    await sb.close()
    import os
    assert not os.path.isdir(ws)


async def test_adopted_workspace_still_confines_paths(tmp_path):
    """接管模式下路径围栏依然生效。"""
    sb = LocalSandbox(workspace=str(tmp_path))
    await sb.start()
    try:
        with pytest.raises(SandboxError):
            await sb.read_file("../../etc/passwd")
    finally:
        await sb.close()
