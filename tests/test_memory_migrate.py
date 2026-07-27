# tests/test_memory_migrate.py
import sqlite3

import sqlite_vec
from sqlite_vec import serialize_float32

from harness.memory.migrate import migrate_memory_db
from harness.memory.record import MemoryFilter
from harness.memory.sqlite_backend import SqliteVecBackend


def _make_old_db(path):
    """造一个旧格式 memory.db：memory_items + memory_vectors。"""
    c = sqlite3.connect(path)
    c.enable_load_extension(True); sqlite_vec.load(c); c.enable_load_extension(False)
    c.execute("CREATE VIRTUAL TABLE memory_vectors USING vec0(embedding float[3])")
    c.execute("CREATE TABLE memory_items(id INTEGER PRIMARY KEY, collection TEXT NOT NULL, "
              "text TEXT NOT NULL, metadata TEXT, created_at TEXT NOT NULL)")
    rows = [("knowledge:u1", "python language", [1.0, 0.0, 0.0]),
            ("conversation:c1", "the cat sat", [0.0, 1.0, 0.0])]
    for coll, text, vec in rows:
        cur = c.execute("INSERT INTO memory_items(collection, text, metadata, created_at) "
                        "VALUES (?, ?, '{}', 'now')", (coll, text))
        c.execute("INSERT INTO memory_vectors(rowid, embedding) VALUES (?, ?)",
                  (cur.lastrowid, serialize_float32(vec)))
    c.commit(); c.close()


def test_migrate_moves_rows(tmp_path):
    old = str(tmp_path / "memory.db")
    _make_old_db(old)
    backend = SqliteVecBackend(":memory:", dimension=3)
    n = migrate_memory_db(old, backend)
    assert n == 2
    # 迁移后可按新隔离/kind 检索
    hits = backend.vector_search([1.0, 0.0, 0.0],
                                 filters=MemoryFilter(owner_id="u1", kind="knowledge"), k=5)
    assert [h.record.text for h in hits] == ["python language"]
    conv = backend.vector_search([0.0, 1.0, 0.0],
                                 filters=MemoryFilter(owner_id="c1", kind="conversation"), k=5)
    assert [h.record.text for h in conv] == ["the cat sat"]


def test_migrate_is_idempotent(tmp_path):
    old = str(tmp_path / "memory.db")
    _make_old_db(old)
    backend = SqliteVecBackend(":memory:", dimension=3)
    assert migrate_memory_db(old, backend) == 2
    assert migrate_memory_db(old, backend) == 0        # 二次迁移不重复
    hits = backend.vector_search([1.0, 0.0, 0.0],
                                 filters=MemoryFilter(owner_id="u1", kind="knowledge"), k=5)
    assert len(hits) == 1                              # 无重复行
