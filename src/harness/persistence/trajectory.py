# src/harness/persistence/trajectory.py
from __future__ import annotations

import json
import sqlite3

from ..events import RunStarted
from ._util import now_iso
from .serialize import event_to_dict


class TrajectoryStore:
    def __init__(self, db_path: str) -> None:
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS trajectory_events("
            "run_id TEXT, seq INTEGER, type TEXT, data TEXT, created_at TEXT, "
            "PRIMARY KEY(run_id, seq))")
        self._conn.commit()

    def append(self, run_id: str, seq: int, event_dict: dict) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO trajectory_events(run_id, seq, type, data, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (run_id, seq, event_dict.get("type", ""),
             json.dumps(event_dict, ensure_ascii=False), now_iso()))
        self._conn.commit()

    def next_seq(self, run_id: str) -> int:
        row = self._conn.execute(
            "SELECT COALESCE(MAX(seq), -1) + 1 FROM trajectory_events WHERE run_id = ?",
            (run_id,)).fetchone()
        return row[0]

    def load(self, run_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT data FROM trajectory_events WHERE run_id = ? ORDER BY seq",
            (run_id,)).fetchall()
        return [json.loads(r[0]) for r in rows]

    def list_run_ids(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT run_id FROM trajectory_events").fetchall()
        return [r[0] for r in rows]

    def delete(self, run_id: str) -> None:
        self._conn.execute("DELETE FROM trajectory_events WHERE run_id = ?", (run_id,))
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


class TrajectorySink:
    """事件流包装器：透传事件，同时按序落库。

    run_id 从 RunStarted 捕获；resume 段不发 RunStarted，调用方需显式传
    `run_id`，此时用 `next_seq` 续号，把恢复段追加到原轨迹之后而不覆盖。
    """

    def __init__(self, store: TrajectoryStore) -> None:
        self._store = store

    async def wrap(self, events, run_id=None):
        seq = self._store.next_seq(run_id) if run_id is not None else 0
        async for ev in events:
            if isinstance(ev, RunStarted):
                run_id = ev.run_id
                seq = self._store.next_seq(run_id)
            if run_id is not None:
                self._store.append(run_id, seq, event_to_dict(ev))
                seq += 1
            yield ev
