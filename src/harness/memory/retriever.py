# src/harness/memory/retriever.py
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .query_planner import QueryPlanner
from .record import MemoryFilter, MemoryRecord
from .reranker import RERANK_SCORE_KEY

log = logging.getLogger("harness.memory.retriever")


def rrf_fuse(ranked_lists: list[list[str]], rrf_k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion：按各路 0-based 排名累加 1/(rrf_k+rank)。无需调参、鲁棒。"""
    scores: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, id_ in enumerate(lst):
            scores[id_] = scores.get(id_, 0.0) + 1.0 / (rrf_k + rank)
    return scores


def recency_score(age_days: float, half_life_days: float) -> float:
    """指数时间衰减，[0,1]。age<=0 记为最新(1.0)，half_life<=0 表示不衰减。"""
    if half_life_days <= 0:
        return 1.0
    return 0.5 ** (max(age_days, 0.0) / half_life_days)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def mmr_select(candidate_ids: list[str], relevance: dict[str, float],
               embeddings: dict[str, list[float]], lambda_: float, k: int) -> list[str]:
    """Maximal Marginal Relevance：贪心选 λ·rel(d) − (1−λ)·max_{s∈已选} cos(d,s)。"""
    selected: list[str] = []
    remaining = list(candidate_ids)
    while remaining and len(selected) < k:
        best, best_score = None, None
        for c in remaining:
            div = 0.0
            if selected:
                div = max(cosine(embeddings.get(c, []), embeddings.get(s, []))
                          for s in selected)
            score = lambda_ * relevance.get(c, 0.0) - (1.0 - lambda_) * div
            if best_score is None or score > best_score:
                best, best_score = c, score
        selected.append(best)
        remaining.remove(best)
    return selected


@dataclass
class ScoredHit:
    record: MemoryRecord
    score: float
    components: dict = field(default_factory=dict)


@dataclass
class RetrievalConfig:
    candidate_pool: int = 20
    w_relevance: float = 1.0
    w_recency: float = 0.2
    w_importance: float = 0.1
    recency_half_life_days: float = 30.0
    use_keyword: bool = True
    use_mmr: bool = True
    mmr_lambda: float = 0.7
    rrf_k: int = 60
    # 查询期召回增强（默认全关=零行为变更）；均需注入 complete，见 QueryPlanner
    use_entity_recall: bool = False
    use_multi_query: bool = False
    use_hyde: bool = False
    multi_query_n: int = 3
    query_plan_timeout_s: float = 2.0
    # 相关性下限：精排分低于它的候选直接丢弃（<=0 关闭）。默认关闭——分数的量纲由具体
    # 精排模型决定（有的输出 [0,1] 概率，有的是未归一化 logit），跨模型套同一个数只会
    # 误伤。装了校准过的精排模型再按实测开启，见 docs/rag-retrieval.md。
    rerank_min_score: float = 0.0


def _age_days(created_at_iso: str, now: datetime) -> float:
    try:
        created = datetime.fromisoformat(created_at_iso)
    except ValueError:
        return 0.0
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return (now - created).total_seconds() / 86400.0


def _minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi <= lo:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class Retriever:
    """在 MemoryBackend 之上编排：向量+关键词召回 → RRF → 加权 → MMR → Reranker → top-k。"""

    def __init__(self, backend, embedder, reranker, config: RetrievalConfig,
                 *, complete=None, now_fn=None) -> None:
        self._backend = backend
        self._embedder = embedder
        self._reranker = reranker
        self._config = config
        self._complete = complete   # 查询期 LLM；None 时三路增强自动跳过
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))

    async def retrieve(self, query_text: str, filters: MemoryFilter, k: int,
                       *, config: RetrievalConfig | None = None) -> list[ScoredHit]:
        cfg = config or self._config
        pool = max(k, cfg.candidate_pool)
        _t0 = time.time()

        # 查询规划：一次 LLM 得多查询改写/HyDE 假设文档/实体键（未启用或失败→空规划降级）
        planner = QueryPlanner(
            self._complete, multi_query=cfg.use_multi_query,
            multi_query_n=cfg.multi_query_n, hyde=cfg.use_hyde,
            entity=cfg.use_entity_recall, timeout_s=cfg.query_plan_timeout_s)
        plan = await planner.plan(query_text)

        # 批量 embedding：原查询 + 各改写 + HyDE 假设文档，一次算完
        embed_texts = [query_text, *plan.variants]
        hyde_idx = -1
        if plan.hypothetical:
            hyde_idx = len(embed_texts)
            embed_texts.append(plan.hypothetical)
        vecs = await self._embedder.embed(embed_texts)

        records: dict[str, MemoryRecord] = {}
        ranked_lists: list[list[str]] = []

        def _absorb(hits) -> None:
            ids: list[str] = []
            for h in hits:
                records.setdefault(h.record.id, h.record)
                ids.append(h.record.id)
            ranked_lists.append(ids)

        _absorb(self._backend.vector_search(vecs[0], filters=filters, k=pool))
        if cfg.use_keyword:
            _absorb(self._backend.keyword_search(query_text, filters=filters, k=pool))
        # 多查询：每条改写的向量各一路
        for i in range(len(plan.variants)):
            _absorb(self._backend.vector_search(vecs[1 + i], filters=filters, k=pool))
        # HyDE：假设文档向量一路
        if hyde_idx >= 0:
            _absorb(self._backend.vector_search(vecs[hyde_idx], filters=filters, k=pool))
        # 实体键：精确取该实体的记忆（需 kind；list_by_entity 返回裸记录，按 recency 排一路）
        if plan.entity_keys and filters.kind:
            for key in plan.entity_keys:
                recs = sorted(self._backend.list_by_entity(filters.owner_id, filters.kind, key),
                              key=lambda r: r.created_at, reverse=True)
                ids = []
                for r in recs:
                    records.setdefault(r.id, r)
                    ids.append(r.id)
                ranked_lists.append(ids)

        if not records:
            return []

        rel = rrf_fuse(ranked_lists, cfg.rrf_k)
        rel_norm = _minmax(rel)
        now = self._now_fn()
        scored: dict[str, tuple[float, dict]] = {}
        for id_, rec in records.items():
            rec_s = recency_score(_age_days(rec.created_at, now), cfg.recency_half_life_days)
            rel_s = rel_norm.get(id_, 0.0)
            final = (cfg.w_relevance * rel_s + cfg.w_recency * rec_s
                     + cfg.w_importance * rec.importance)
            scored[id_] = (final, {"relevance": rel_s, "recency": rec_s,
                                   "importance": rec.importance})

        order = sorted(records.keys(), key=lambda i: scored[i][0], reverse=True)
        if cfg.use_mmr:
            embs = self._backend.get_embeddings(order)
            order = mmr_select(order, {i: scored[i][0] for i in order},
                               embs, cfg.mmr_lambda, len(order))

        hits = [ScoredHit(record=records[i], score=scored[i][0], components=scored[i][1])
                for i in order]
        hits = await self._reranker.rerank(query_text, hits)
        dropped = 0
        if cfg.rerank_min_score > 0:
            # 只筛「精排真给过分」的候选：端点故障时 rerank 降级为原序返回、components 里
            # 没有分数，此时一条都不该丢——宁可放行不相关的，也不能因精排挂了就把知识库
            # 整个判空，那会让用户以为资料没存进去。
            kept = [h for h in hits
                    if h.components.get(RERANK_SCORE_KEY) is None
                    or h.components[RERANK_SCORE_KEY] >= cfg.rerank_min_score]
            dropped = len(hits) - len(kept)
            if dropped and not kept:
                # 全军覆没有两种可能：本次查询确实与库无关（正常），或换了精排模型后阈值
                # 量纲对不上（故障）。两者在结果上完全一样——都是「知识库看起来空的」。
                # 不记一笔的话，后者会静默劣化：用户只会觉得资料没存进去。
                log.warning("检索相关性下限滤掉了全部 %d 条候选（阈值=%.2f，最高分=%.3f）。"
                            "若换过精排模型，请按新模型实测重新标定 rerank_min_score。",
                            dropped, cfg.rerank_min_score,
                            max(h.components.get(RERANK_SCORE_KEY, 0.0) for h in hits))
            hits = kept
        log.info("检索 query=%d字 路数=%d 候选=%d 低相关丢弃=%d 返回=%d "
                 "增强(改写=%d hyde=%s 实体=%d) 耗时%dms",
                 len(query_text or ""), len(ranked_lists), len(records), dropped,
                 min(k, len(hits)),
                 len(plan.variants), bool(plan.hypothetical), len(plan.entity_keys),
                 round((time.time() - _t0) * 1000))
        return hits[:k]
