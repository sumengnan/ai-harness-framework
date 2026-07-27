# src/harness/memory/maintainer.py
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from .record import MemType, MemoryRecord
from .retriever import cosine

log = logging.getLogger(__name__)

_DISTILL_SYS = (
    "你是记忆固化器。下面是同一主题的若干条零散情景记忆，请把它们蒸馏合并成"
    "**一条**简洁、稳定的语义事实（中文，一句话）。只输出这条事实文本，不要解释、不要列表。"
)

# semantic 偏好整理：把同主题的多条用户偏好/事实合并成一条，不丢关键信息（手动触发）。
_DISTILL_SYS_SEMANTIC = (
    "你在整理用户的长期偏好库。下面是同主题的若干条零散偏好/事实，请把它们合并成"
    "**一条**简洁、稳定、不遗漏关键信息的偏好陈述（中文，一句话）。"
    "只输出这条陈述文本，不要解释、不要列表。"
)


@dataclass
class ConsolidationConfig:
    sim_threshold: float = 0.85     # 贪心聚类的平均余弦相似度阈值
    min_cluster: int = 2            # 成簇最小成员数
    max_source: int = 200           # 单次 consolidate 处理的 episodic 上限


class MemoryMaintainer:
    """按需记忆维护：过期剔除 + consolidation（episodic 蒸馏为 semantic）。"""

    def __init__(self, backend, embedder, complete,
                 config: ConsolidationConfig, *, now_fn=None) -> None:
        self._backend = backend
        self._embedder = embedder
        self._complete = complete
        self._config = config
        self._now_fn = now_fn or (lambda: int(time.time()))

    def _cluster(self, records: list[MemoryRecord],
                 embs: dict[str, list[float]]) -> list[list[MemoryRecord]]:
        clusters: list[list[MemoryRecord]] = []
        for r in records:
            v = embs.get(r.id)
            if v is None:
                continue
            placed = False
            for cl in clusters:
                sims = [cosine(v, embs[m.id]) for m in cl if m.id in embs]
                if sims and sum(sims) / len(sims) >= self._config.sim_threshold:
                    cl.append(r)
                    placed = True
                    break
            if not placed:
                clusters.append([r])
        return clusters

    async def consolidate(self, owner_id: str, kind: str, *,
                          mem_type: MemType = MemType.EPISODIC,
                          distill_sys: str | None = None) -> dict:
        """把 owner+kind 下同 mem_type、语义相近的一簇蒸馏合并成一条 semantic，原簇标记取代。

        mem_type=EPISODIC（默认）：情景经验蒸馏为语义事实（后台自动维护用）。
        mem_type=SEMANTIC：把同主题的多条偏好合并成一条（手动「整理偏好」用，见 consolidate_semantic）。
        """
        src = [r for r in self._backend.list_by_owner(owner_id, kind)
               if r.mem_type == mem_type]
        records = src[:self._config.max_source]
        if len(src) > self._config.max_source:
            log.info("consolidate(%s): %d 条超过 max_source=%d，本次只处理前 %d 条",
                     mem_type.value, len(src), self._config.max_source, self._config.max_source)
        if not records:
            return {"clusters": 0, "merged": 0, "created": 0}
        embs = self._backend.get_embeddings([r.id for r in records])
        clusters = self._cluster(records, embs)
        _distill = distill_sys or _DISTILL_SYS
        nclusters = merged = created = 0
        for cl in clusters:
            if len(cl) < self._config.min_cluster:
                continue
            try:
                text = (await self._complete(
                    _distill, "\n".join(m.text for m in cl))).strip()
            except Exception as e:
                log.warning("consolidate distill failed: %s", e)
                continue
            if not text:
                continue
            try:
                vec = (await self._embedder.embed([text]))[0]
                new_ids = self._backend.upsert([MemoryRecord(
                    owner_id=owner_id, kind=kind, mem_type=MemType.SEMANTIC,
                    text=text, embedding=vec, source="consolidate")])
            except Exception as e:
                log.warning("consolidate upsert failed: %s", e)
                continue
            try:
                self._backend.set_superseded([m.id for m in cl])
            except Exception as e:                       # 补偿回滚：删掉刚写入的新记录，保持原子
                log.warning("consolidate supersede failed, rolling back: %s", e)
                self._backend.delete(new_ids)
                continue
            nclusters += 1
            merged += len(cl)
            created += 1
        return {"clusters": nclusters, "merged": merged, "created": created}

    async def consolidate_semantic(self, owner_id: str, kind: str) -> dict:
        """把同主题的多条 semantic 偏好合并成一条（手动触发的「整理相似偏好」）。"""
        return await self.consolidate(owner_id, kind, mem_type=MemType.SEMANTIC,
                                      distill_sys=_DISTILL_SYS_SEMANTIC)

    async def maintain(self, owner_id: str, kind: str) -> dict:
        purged = self._backend.purge_expired(self._now_fn())
        result = await self.consolidate(owner_id, kind)
        return {"purged": purged, **result}
