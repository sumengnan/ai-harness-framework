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
