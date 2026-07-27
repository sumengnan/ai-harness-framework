# src/harness/tools/builtins/_sandbox_util.py
from __future__ import annotations

from ...sandbox.base import ExecResult


def truncate(text: str, max_chars: int) -> str:
    if max_chars and len(text) > max_chars:
        return text[:max_chars] + "…(已截断)"
    return text


def format_exec(res: ExecResult, max_chars: int) -> str:
    header = f"exit_code={res.exit_code}" + ("（超时）" if res.timed_out else "")
    parts = [header]
    if res.stdout:
        parts.append("stdout:\n" + res.stdout)
    if res.stderr:
        parts.append("stderr:\n" + res.stderr)
    return truncate("\n".join(parts), max_chars)
