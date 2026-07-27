from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

import sqlite_vec


@dataclass
class MemoryHit:
    text: str
    collection: str
    metadata: dict
    distance: float
    id: str = ""            # 命中记录的 id（facade 回填；旧 MemoryStore 路径留空）
    created_at: str = ""    # 命中记录的创建时间（同上）
    rerank_score: float | None = None
    # 精排（qwen3-rerank）算出的绝对相关性分，量纲 [0,1]（实测 unrelated≈0.26、related≈0.43）。
    # 这是整条检索链上唯一没被 RRF/minmax 抹掉的绝对信号：distance 源自候选集内 minmax 归一化的
    # 融合分，最高的那条恒为 1.0，只反映相对排名、不反映「到底有多相关」。
    # 为 None 表示本次没有精排分（精排关闭，或精排端点降级为原序返回）。


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


_OVERFETCH_MULTIPLIER = 4


class MemoryStore:
    """sqlite-vec 向量存储。向量表 rowid 与元数据表 id 对齐。"""

    def __init__(self, db_path: str, dimension: int) -> None:
        self._dim = dimension
        self._conn = sqlite3.connect(db_path)
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        self._conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS memory_vectors "
            f"USING vec0(embedding float[{dimension}])"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS memory_items("
            "id INTEGER PRIMARY KEY, collection TEXT NOT NULL, text TEXT NOT NULL, "
            "metadata TEXT, created_at TEXT NOT NULL)"
        )
        self._conn.commit()

    def add(self, items: list[tuple[str, str, dict, list[float]]]) -> list[int]:
        ids: list[int] = []
        for collection, text, metadata, embedding in items:
            if len(embedding) != self._dim:
                raise ValueError(
                    f"向量维度不符：期望 {self._dim}，收到 {len(embedding)}"
                )
            cur = self._conn.execute(
                "INSERT INTO memory_items(collection, text, metadata, created_at) "
                "VALUES (?, ?, ?, ?)",
                (collection, text, json.dumps(metadata or {}), _now()),
            )
            rowid = cur.lastrowid
            self._conn.execute(
                "INSERT INTO memory_vectors(rowid, embedding) VALUES (?, ?)",
                (rowid, sqlite_vec.serialize_float32(embedding)),
            )
            ids.append(rowid)
        self._conn.commit()
        return ids

    def search(self, collection: str, query_embedding: list[float], k: int) -> list[MemoryHit]:
        total = self._conn.execute(
            "SELECT COUNT(*) FROM memory_vectors"
        ).fetchone()[0]
        if total == 0:
            return []
        serialized = sqlite_vec.serialize_float32(query_embedding)
        # 自适应 over-fetch：其他 collection 可能挤占前排，命中不足时逐步扩大
        fetch = min(total, max(k, 1) * _OVERFETCH_MULTIPLIER)
        hits: list[MemoryHit] = []
        while True:
            hits = self._knn_filtered(collection, serialized, fetch, k)
            if len(hits) >= k or fetch >= total:
                break
            fetch = min(total, fetch * _OVERFETCH_MULTIPLIER)
        return hits

    def _knn_filtered(
        self, collection: str, serialized: bytes, fetch: int, k: int
    ) -> list[MemoryHit]:
        rows = self._conn.execute(
            "SELECT rowid, distance FROM memory_vectors "
            "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (serialized, fetch),
        ).fetchall()
        hits: list[MemoryHit] = []
        for rowid, distance in rows:
            item = self._conn.execute(
                "SELECT collection, text, metadata FROM memory_items WHERE id = ?",
                (rowid,),
            ).fetchone()
            if item is None:
                continue
            coll, text, metadata = item
            if coll != collection:
                continue
            hits.append(MemoryHit(
                text=text, collection=coll,
                metadata=json.loads(metadata or "{}"), distance=distance))
            if len(hits) >= k:
                break
        return hits

    def delete(self, ids: list[int]) -> None:
        for i in ids:
            self._conn.execute("DELETE FROM memory_items WHERE id = ?", (i,))
            self._conn.execute("DELETE FROM memory_vectors WHERE rowid = ?", (i,))
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
