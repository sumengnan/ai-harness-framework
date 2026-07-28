# src/harness/persistence/checkpoint.py
from __future__ import annotations

import json
import sqlite3
import threading

from ..state import RunState
from ._util import now_iso
from .serialize import runstate_from_dict, runstate_to_dict


class CheckpointStore:
    """RunState 快照存取，供断点续跑（resume）。

    已知限制：快照在步边界（StepFinished 之后）保存，resume 从上一个完整步的
    下一步重跑。若中断发生在某步执行到一半，该步会被整步重跑——其中的**有副作用
    工具**（如 write_file / run_shell）可能被**重复执行**。调用方需保证工具幂等，
    或自行去重。

    线程安全：连接以 check_same_thread=False 建立，并由一把锁串行化所有读写，
    因此可以在多个线程/事件循环里共用同一个实例。sqlite3 的默认连接只允许创建它
    的那个线程使用，而 agent 常跑在线程池或另起的事件循环里，共用实例会直接抛
    ProgrammingError——这不是理论风险。
    """

    def __init__(self, db_path: str) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS checkpoints("
                "run_id TEXT PRIMARY KEY, state TEXT, step INTEGER, updated_at TEXT)")
            self._conn.commit()

    def save(self, state: RunState) -> None:
        payload = json.dumps(runstate_to_dict(state), ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO checkpoints(run_id, state, step, updated_at) VALUES (?, ?, ?, ?)",
                (state.run_id, payload, state.step, now_iso()))
            self._conn.commit()

    def load(self, run_id: str) -> RunState | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT state FROM checkpoints WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return runstate_from_dict(json.loads(row[0]))

    def delete(self, run_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM checkpoints WHERE run_id = ?", (run_id,))
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
