import os
import pytest
from harness.sandbox.base import resolve_in_workspace, SandboxError, ExecResult


def test_resolve_ok(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    p = resolve_in_workspace(str(ws), "sub/file.txt")
    assert p == os.path.join(os.path.realpath(str(ws)), "sub", "file.txt")


def test_resolve_dot_is_workspace(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    assert resolve_in_workspace(str(ws), ".") == os.path.realpath(str(ws))


def test_resolve_relative_escape(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    with pytest.raises(SandboxError):
        resolve_in_workspace(str(ws), "../../etc/passwd")


def test_resolve_absolute_escape(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    with pytest.raises(SandboxError):
        resolve_in_workspace(str(ws), "/etc/passwd")


def test_resolve_symlink_escape(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    secret = tmp_path / "secret.txt"; secret.write_text("x")
    (ws / "link").symlink_to(secret)
    with pytest.raises(SandboxError):
        resolve_in_workspace(str(ws), "link")


def test_exec_result_defaults():
    r = ExecResult("o", "e", 0)
    assert r.timed_out is False
