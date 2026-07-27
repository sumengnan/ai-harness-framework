# src/harness/memory/migrate.py
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

import sqlite_vec

from .backend import MemoryBackend
from .memory import collection_to_scope
from .record import MemoryRecord, MemType

_MARK = "sp1_kernel"


def migrate_memory_db(old_path: str, backend: MemoryBackend) -> int:
    """把旧 memory.db 的行迁入 backend。幂等：已迁移返回 0。返回迁移条数。"""
    conn = sqlite3.connect(old_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("CREATE TABLE IF NOT EXISTS _memory_migrations("
                 "name TEXT PRIMARY KEY, done_at TEXT)")
    if conn.execute("SELECT 1 FROM _memory_migrations WHERE name = ?", (_MARK,)).fetchone():
        conn.close()
        return 0
    rows = conn.execute(
        "SELECT id, collection, text, metadata FROM memory_items").fetchall()
    records: list[MemoryRecord] = []
    for old_id, collection, text, meta in rows:
        emb = conn.execute(
            "SELECT vec_to_json(embedding) FROM memory_vectors WHERE rowid = ?",
            (old_id,)).fetchone()
        if emb is None:
            continue
        owner_id, kind = collection_to_scope(collection)
        records.append(MemoryRecord(
            owner_id=owner_id, kind=kind, mem_type=MemType.SEMANTIC, text=text,
            embedding=json.loads(emb[0]), metadata=json.loads(meta or "{}"),
            source="migrated:sp1"))
    backend.upsert(records)
    conn.execute("INSERT INTO _memory_migrations(name, done_at) VALUES (?, ?)",
                 (_MARK, datetime.now(timezone.utc).isoformat()))
    conn.commit()
    conn.close()
    return len(records)
