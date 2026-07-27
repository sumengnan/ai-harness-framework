# src/harness/memory/eval.py
from __future__ import annotations

from dataclasses import dataclass

from .record import MemoryFilter


@dataclass
class GoldenCase:
    query: str
    relevant_ids: set[str]


async def evaluate(retriever, filters: MemoryFilter,
                   cases: list[GoldenCase], k: int) -> dict:
    """在 golden set 上算 hit@k 与 MRR。retriever 需实现 async retrieve(query, filters, k)。"""
    hit_count = 0
    rr_sum = 0.0
    for case in cases:
        results = await retriever.retrieve(case.query, filters, k)
        ids = [h.record.id for h in results]
        if any(i in case.relevant_ids for i in ids):
            hit_count += 1
        for rank, i in enumerate(ids):
            if i in case.relevant_ids:
                rr_sum += 1.0 / (rank + 1)
                break
    n = len(cases) or 1
    return {"hit@k": hit_count / n, "mrr": rr_sum / n}


def store_size(backend, owner_id: str, kind: str) -> int:
    """命名空间内未废弃记录数（评测「固化降噪」用）。"""
    return backend.count_by_owner(owner_id, kind)
