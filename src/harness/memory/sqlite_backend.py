# src/harness/memory/sqlite_backend.py
from __future__ import annotations

import json
import sqlite3
import time

import sqlite_vec
from sqlite_vec import serialize_float32

from .record import MemoryFilter, MemoryHit, MemoryRecord, MemType

# memory_records 的列顺序（rowid 之外），供 INSERT / 回读复用
_COLS = ("id", "owner_id", "kind", "mem_type", "text", "entity_key", "version",
         "superseded", "importance", "created_at", "last_accessed_at",
         "access_count", "expires_at", "source", "metadata")


class SqliteVecBackend:
    """MemoryBackend 的 sqlite-vec 实现：vec0 过滤表 + companion 数据表，rowid 对齐。"""

    def __init__(self, db_path: str, dimension: int, *, now_fn=None) -> None:
        self._dim = dimension
        self._now_fn = now_fn or (lambda: int(time.time()))
        self._conn = sqlite3.connect(db_path)
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        self._conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec USING vec0("
            f"owner_id TEXT partition key, kind TEXT, mem_type TEXT, "
            f"superseded INTEGER, expires_at INTEGER, embedding float[{dimension}])")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS memory_records("
            "rowid INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL, "
            "owner_id TEXT NOT NULL, kind TEXT NOT NULL, mem_type TEXT NOT NULL, "
            "text TEXT NOT NULL, entity_key TEXT NOT NULL DEFAULT '', "
            "version INTEGER NOT NULL DEFAULT 1, superseded INTEGER NOT NULL DEFAULT 0, "
            "importance REAL NOT NULL DEFAULT 0.5, created_at TEXT NOT NULL, "
            "last_accessed_at TEXT NOT NULL, access_count INTEGER NOT NULL DEFAULT 0, "
            "expires_at INTEGER NOT NULL DEFAULT 0, source TEXT NOT NULL DEFAULT '', "
            "metadata TEXT NOT NULL DEFAULT '{}')")
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_mem_entity "
            "ON memory_records(owner_id, kind, entity_key)")
        self._conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts "
            "USING fts5(text, tokenize='trigram')")
        self._conn.commit()

    # ---- 写 ----
    def upsert(self, records: list[MemoryRecord]) -> list[str]:
        for r in records:                       # 整批预校验，保证异常时零写入（原子）
            if len(r.embedding) != self._dim:
                raise ValueError(f"向量维度不符：期望 {self._dim}，收到 {len(r.embedding)}")
        ids: list[str] = []
        for r in records:
            old = self._conn.execute(
                "SELECT rowid FROM memory_records WHERE id = ?", (r.id,)).fetchone()
            if old is not None:
                self._delete_rowid(old[0])            # 覆盖：先删旧行（两表）
            cur = self._conn.execute(
                f"INSERT INTO memory_records({','.join(_COLS)}) "
                f"VALUES ({','.join('?' * len(_COLS))})",
                (r.id, r.owner_id, r.kind, r.mem_type.value, r.text, r.entity_key,
                 r.version, r.superseded, r.importance, r.created_at,
                 r.last_accessed_at, r.access_count, r.expires_at, r.source,
                 json.dumps(r.metadata, ensure_ascii=False)))
            rowid = cur.lastrowid
            self._conn.execute(
                "INSERT INTO memory_vec(rowid, owner_id, kind, mem_type, superseded, "
                "expires_at, embedding) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (rowid, r.owner_id, r.kind, r.mem_type.value, r.superseded,
                 r.expires_at, serialize_float32(r.embedding)))
            self._conn.execute(
                "INSERT INTO memory_fts(rowid, text) VALUES (?, ?)", (rowid, r.text))
            ids.append(r.id)
        self._conn.commit()
        return ids

    def delete(self, ids: list[str]) -> None:
        for i in ids:
            row = self._conn.execute(
                "SELECT rowid FROM memory_records WHERE id = ?", (i,)).fetchone()
            if row is not None:
                self._delete_rowid(row[0])
        self._conn.commit()

    def _delete_rowid(self, rowid: int) -> None:
        self._conn.execute("DELETE FROM memory_records WHERE rowid = ?", (rowid,))
        self._conn.execute("DELETE FROM memory_vec WHERE rowid = ?", (rowid,))
        self._conn.execute("DELETE FROM memory_fts WHERE rowid = ?", (rowid,))

    def set_superseded(self, ids: list[str]) -> None:
        for i in ids:
            row = self._conn.execute(
                "SELECT rowid FROM memory_records WHERE id = ?", (i,)).fetchone()
            if row is None:
                continue
            self._conn.execute(
                "UPDATE memory_records SET superseded = 1 WHERE rowid = ?", (row[0],))
            self._conn.execute(
                "UPDATE memory_vec SET superseded = 1 WHERE rowid = ?", (row[0],))
        self._conn.commit()

    def purge_expired(self, now: int) -> int:
        rows = self._conn.execute(
            "SELECT rowid FROM memory_records WHERE expires_at != 0 AND expires_at <= ?",
            (now,)).fetchall()
        for (rowid,) in rows:
            self._conn.execute("DELETE FROM memory_records WHERE rowid = ?", (rowid,))
            self._conn.execute("DELETE FROM memory_vec WHERE rowid = ?", (rowid,))
            self._conn.execute("DELETE FROM memory_fts WHERE rowid = ?", (rowid,))
        self._conn.commit()
        return len(rows)

    # ---- 读 ----
    def get(self, ids: list[str]) -> list[MemoryRecord]:
        out: list[MemoryRecord] = []
        for i in ids:
            row = self._conn.execute(
                f"SELECT {','.join(_COLS)} FROM memory_records WHERE id = ?", (i,)).fetchone()
            if row is not None:
                out.append(self._row_to_record(row))
        return out

    def list_by_owner(self, owner_id: str, kind: str, *,
                      limit: int | None = None, offset: int = 0) -> list[MemoryRecord]:
        """按 owner+kind 分页列举未废弃记录（rowid 降序 = 最近写入在前）。"""
        sql = (f"SELECT {','.join(_COLS)} FROM memory_records "
               "WHERE owner_id = ? AND kind = ? AND superseded = 0 ORDER BY rowid DESC")
        params: list = [owner_id, kind]
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params += [limit, offset]
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_record(r) for r in rows]

    def count_by_owner(self, owner_id: str, kind: str, *,
                       mem_type: "MemType | str | None" = None) -> int:
        sql = ("SELECT COUNT(*) FROM memory_records "
               "WHERE owner_id = ? AND kind = ? AND superseded = 0")
        params: list = [owner_id, kind]
        if mem_type is not None:
            sql += " AND mem_type = ?"
            params.append(mem_type.value if isinstance(mem_type, MemType) else str(mem_type))
        return self._conn.execute(sql, params).fetchone()[0]

    def list_by_entity(self, owner_id: str, kind: str,
                       entity_key: str) -> list[MemoryRecord]:
        rows = self._conn.execute(
            f"SELECT {','.join(_COLS)} FROM memory_records "
            "WHERE owner_id = ? AND kind = ? AND entity_key = ? AND superseded = 0",
            (owner_id, kind, entity_key)).fetchall()
        return [self._row_to_record(r) for r in rows]

    def vector_search(self, query_embedding: list[float], *,
                      filters: MemoryFilter, k: int) -> list[MemoryHit]:
        conds = ["embedding MATCH ?", "k = ?", "owner_id = ?"]
        params: list = [serialize_float32(query_embedding), k, filters.owner_id]
        if filters.kind is not None:
            conds.append("kind = ?"); params.append(filters.kind)
        if filters.mem_type is not None:
            conds.append("mem_type = ?"); params.append(filters.mem_type)
        if not filters.include_superseded:
            conds.append("superseded = ?"); params.append(0)
        if not filters.include_expired:
            conds.append("(expires_at = 0 OR expires_at > ?)")
            params.append(self._now_fn())
        sql = ("SELECT rowid, distance FROM memory_vec WHERE "
               + " AND ".join(conds) + " ORDER BY distance")
        rows = self._conn.execute(sql, params).fetchall()
        hits: list[MemoryHit] = []
        for rowid, distance in rows:
            rec = self._conn.execute(
                f"SELECT {','.join(_COLS)} FROM memory_records WHERE rowid = ?",
                (rowid,)).fetchone()
            if rec is not None:
                hits.append(MemoryHit(record=self._row_to_record(rec), distance=distance))
        return hits

    def keyword_search(self, query_text: str, *,
                       filters: MemoryFilter, k: int) -> list[MemoryHit]:
        if not query_text or not query_text.strip():
            return []
        match_query = '"' + query_text.replace('"', '""') + '"'   # 作为短语匹配，避开 FTS5 特殊字符语法
        conds = ["memory_fts MATCH ?", "r.owner_id = ?"]
        params: list = [match_query, filters.owner_id]
        if filters.kind is not None:
            conds.append("r.kind = ?"); params.append(filters.kind)
        if filters.mem_type is not None:
            conds.append("r.mem_type = ?"); params.append(filters.mem_type)
        if not filters.include_superseded:
            conds.append("r.superseded = ?"); params.append(0)
        if not filters.include_expired:
            conds.append("(r.expires_at = 0 OR r.expires_at > ?)")
            params.append(self._now_fn())
        cols = ",".join("r." + c for c in _COLS)
        sql = (f"SELECT {cols}, bm25(memory_fts) AS score "
               "FROM memory_fts JOIN memory_records r ON r.rowid = memory_fts.rowid "
               "WHERE " + " AND ".join(conds) + " ORDER BY score LIMIT ?")
        params.append(k)
        try:
            rows = self._conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            return []
        hits: list[MemoryHit] = []
        for row in rows:
            rec = self._row_to_record(row[:len(_COLS)])
            hits.append(MemoryHit(record=rec, distance=row[len(_COLS)]))
        return hits

    def get_embeddings(self, ids: list[str]) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for i in ids:
            row = self._conn.execute(
                "SELECT rowid FROM memory_records WHERE id = ?", (i,)).fetchone()
            if row is None:
                continue
            emb = self._conn.execute(
                "SELECT vec_to_json(embedding) FROM memory_vec WHERE rowid = ?",
                (row[0],)).fetchone()
            if emb is not None:
                out[i] = json.loads(emb[0])
        return out

    def _row_to_record(self, row: tuple) -> MemoryRecord:
        d = dict(zip(_COLS, row))
        return MemoryRecord(
            owner_id=d["owner_id"], kind=d["kind"], mem_type=MemType(d["mem_type"]),
            text=d["text"], embedding=[], id=d["id"], entity_key=d["entity_key"],
            version=d["version"], superseded=d["superseded"], importance=d["importance"],
            created_at=d["created_at"], last_accessed_at=d["last_accessed_at"],
            access_count=d["access_count"], expires_at=d["expires_at"],
            source=d["source"], metadata=json.loads(d["metadata"] or "{}"))

    def close(self) -> None:
        self._conn.close()
