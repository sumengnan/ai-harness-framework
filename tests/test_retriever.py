# tests/test_retriever.py
from datetime import datetime, timedelta, timezone

from harness.memory.record import MemType, MemoryFilter, MemoryRecord
from harness.memory.reranker import NoOpReranker
from harness.memory.retriever import RetrievalConfig, Retriever, ScoredHit
from harness.memory.sqlite_backend import SqliteVecBackend

_NOW = datetime(2026, 7, 11, tzinfo=timezone.utc)


def _iso(days_ago):
    return (_NOW - timedelta(days=days_ago)).isoformat()


def _rec(text, vec, rid, importance=0.5, days_ago=0):
    r = MemoryRecord(owner_id="u1", kind="k", mem_type=MemType.SEMANTIC,
                     text=text, embedding=vec, id=rid, importance=importance)
    r.created_at = _iso(days_ago)
    return r


def _retriever(backend, embedder, config, reranker=None):
    return Retriever(backend, embedder, reranker or NoOpReranker(), config,
                     now_fn=lambda: _NOW)


async def test_retrieve_returns_scored_hits(mock_embedder):
    emb = mock_embedder(dimension=64)
    b = SqliteVecBackend(":memory:", dimension=64)
    for rid, text in [("a", "python programming"), ("b", "the cat sat")]:
        v = (await emb.embed([text]))[0]
        b.upsert([_rec(text, v, rid)])
    cfg = RetrievalConfig(use_keyword=False, use_mmr=False, w_recency=0.0, w_importance=0.0)
    r = _retriever(b, emb, cfg)
    hits = await r.retrieve("cat", MemoryFilter(owner_id="u1"), k=1)
    assert isinstance(hits[0], ScoredHit)
    assert hits[0].record.text == "the cat sat"


async def test_recency_breaks_ties(mock_embedder):
    emb = mock_embedder(dimension=64)
    b = SqliteVecBackend(":memory:", dimension=64)
    v = (await emb.embed(["排序算法"]))[0]
    b.upsert([_rec("排序算法", v, "old", days_ago=60),
              _rec("排序算法", v, "new", days_ago=0)])
    cfg = RetrievalConfig(use_keyword=False, use_mmr=False,
                          w_relevance=1.0, w_recency=1.0, w_importance=0.0,
                          recency_half_life_days=30.0)
    r = _retriever(b, emb, cfg)
    hits = await r.retrieve("排序算法", MemoryFilter(owner_id="u1"), k=2)
    assert hits[0].record.id == "new"


async def test_mmr_diversifies(mock_embedder):
    emb = mock_embedder(dimension=64)
    b = SqliteVecBackend(":memory:", dimension=64)
    va = (await emb.embed(["快速排序快速排序"]))[0]
    vc = (await emb.embed(["图论最短路径"]))[0]
    b.upsert([_rec("快速排序快速排序", va, "a", days_ago=0),
              _rec("快速排序快速排序", va, "b", days_ago=0),
              _rec("图论最短路径", vc, "c", days_ago=0)])
    cfg = RetrievalConfig(use_keyword=False, use_mmr=True, mmr_lambda=0.5,
                          w_recency=0.0, w_importance=0.0)
    r = _retriever(b, emb, cfg)
    ids = [h.record.id for h in await r.retrieve("快速排序快速排序",
                                                 MemoryFilter(owner_id="u1"), k=2)]
    assert "c" in ids


async def test_reranker_is_applied(mock_embedder):
    emb = mock_embedder(dimension=64)
    b = SqliteVecBackend(":memory:", dimension=64)
    for rid, t in [("a", "aaa bbb"), ("b", "ccc ddd")]:
        v = (await emb.embed([t]))[0]
        b.upsert([_rec(t, v, rid)])

    class ReverseReranker:
        async def rerank(self, query, candidates):
            return list(reversed(candidates))

    cfg = RetrievalConfig(use_keyword=False, use_mmr=False)
    base = [h.record.id for h in await _retriever(b, emb, cfg).retrieve(
        "aaa bbb", MemoryFilter(owner_id="u1"), k=2)]
    rev = [h.record.id for h in await _retriever(b, emb, cfg, ReverseReranker()).retrieve(
        "aaa bbb", MemoryFilter(owner_id="u1"), k=2)]
    assert rev == list(reversed(base))


async def test_hybrid_retrieve_excludes_expired(mock_embedder):
    # 默认 use_keyword=True（hybrid 检索），过期记录不应通过关键词路径泄漏
    emb = mock_embedder(dimension=64)
    b = SqliteVecBackend(":memory:", dimension=64, now_fn=lambda: 1000)
    v = (await emb.embed(["排序算法讲解"]))[0]
    rec = _rec("排序算法讲解", v, "expired")
    rec.expires_at = 100  # 已过期（<= now=1000）
    b.upsert([rec])
    cfg = RetrievalConfig()  # 默认 use_keyword=True
    r = Retriever(b, emb, NoOpReranker(), cfg, now_fn=lambda: _NOW)
    hits = await r.retrieve("排序算法讲解", MemoryFilter(owner_id="u1"), k=5)
    assert hits == []


async def test_k_larger_than_pool_not_truncated(mock_embedder):
    emb = mock_embedder(dimension=64)
    b = SqliteVecBackend(":memory:", dimension=64)
    for i in range(30):
        v = (await emb.embed([f"item number {i}"]))[0]
        b.upsert([_rec(f"item number {i}", v, f"r{i}")])
    cfg = RetrievalConfig(candidate_pool=20, use_keyword=False, use_mmr=False,
                          w_recency=0.0, w_importance=0.0)
    hits = await _retriever(b, emb, cfg).retrieve(
        "item number", MemoryFilter(owner_id="u1"), k=25)
    assert len(hits) == 25          # 不被 candidate_pool=20 截断


# ---- 查询期召回增强：实体键 / 多查询 / HyDE ----

class _Comp:
    """假 LLM completer，返回预设的规划 JSON。"""
    def __init__(self, resp):
        self.resp = resp

    async def __call__(self, system, user):
        return self.resp


async def test_entity_recall_surfaces_record_filtered_out_of_base(mock_embedder):
    # b 是 episodic，被 mem_type=semantic 过滤挡在基础召回之外；实体键路（list_by_entity
    # 不认 mem_type 过滤）应把它捞回来。
    emb = mock_embedder(dimension=64)
    backend = SqliteVecBackend(":memory:", dimension=64)
    va = (await emb.embed(["语义A"]))[0]
    vb = (await emb.embed(["情节B"]))[0]
    backend.upsert([
        MemoryRecord(owner_id="u1", kind="k", mem_type=MemType.SEMANTIC,
                     text="语义A", embedding=va, id="a"),
        MemoryRecord(owner_id="u1", kind="k", mem_type=MemType.EPISODIC,
                     text="情节B", embedding=vb, id="b", entity_key="user.pref.x"),
    ])
    filt = MemoryFilter(owner_id="u1", kind="k", mem_type="semantic")
    base = RetrievalConfig(use_keyword=False, use_mmr=False)
    off = await _retriever(backend, emb, base).retrieve("q", filt, k=5)
    assert "b" not in {h.record.id for h in off}          # 基础召回看不到 episodic 的 b

    on_cfg = RetrievalConfig(use_keyword=False, use_mmr=False, use_entity_recall=True)
    r_on = Retriever(backend, emb, NoOpReranker(), on_cfg,
                     complete=_Comp('{"entity_keys":["user.pref.x"]}'), now_fn=lambda: _NOW)
    on = await r_on.retrieve("q", filt, k=5)
    assert "b" in {h.record.id for h in on}               # 实体键路把 b 召回


async def test_multi_query_recalls_doc_missed_by_base(mock_embedder):
    # candidate_pool=1：基础只召回离原查询最近的 a；改写命中 b。b importance 更高，k=1 时靠前。
    emb = mock_embedder(dimension=64)
    backend = SqliteVecBackend(":memory:", dimension=64)
    va = (await emb.embed(["原始问题"]))[0]
    vb = (await emb.embed(["改写命中"]))[0]
    backend.upsert([
        MemoryRecord(owner_id="u1", kind="k", mem_type=MemType.SEMANTIC,
                     text="原始问题", embedding=va, id="a", importance=0.1),
        MemoryRecord(owner_id="u1", kind="k", mem_type=MemType.SEMANTIC,
                     text="改写命中", embedding=vb, id="b", importance=0.9),
    ])
    filt = MemoryFilter(owner_id="u1", kind="k")
    base = RetrievalConfig(candidate_pool=1, use_keyword=False, use_mmr=False)
    off = await _retriever(backend, emb, base).retrieve("原始问题", filt, k=1)
    assert {h.record.id for h in off} == {"a"}            # 基础 pool=1 只召回 a

    on_cfg = RetrievalConfig(candidate_pool=1, use_keyword=False, use_mmr=False,
                             use_multi_query=True)
    r_on = Retriever(backend, emb, NoOpReranker(), on_cfg,
                     complete=_Comp('{"variants":["改写命中"]}'), now_fn=lambda: _NOW)
    on = await r_on.retrieve("原始问题", filt, k=1)
    assert "b" in {h.record.id for h in on}               # 改写把 b 召回并靠前


async def test_hyde_recalls_doc_missed_by_base(mock_embedder):
    emb = mock_embedder(dimension=64)
    backend = SqliteVecBackend(":memory:", dimension=64)
    va = (await emb.embed(["原始问题"]))[0]
    vb = (await emb.embed(["假设答案"]))[0]
    backend.upsert([
        MemoryRecord(owner_id="u1", kind="k", mem_type=MemType.SEMANTIC,
                     text="原始问题", embedding=va, id="a", importance=0.1),
        MemoryRecord(owner_id="u1", kind="k", mem_type=MemType.SEMANTIC,
                     text="假设答案", embedding=vb, id="b", importance=0.9),
    ])
    filt = MemoryFilter(owner_id="u1", kind="k")
    on_cfg = RetrievalConfig(candidate_pool=1, use_keyword=False, use_mmr=False, use_hyde=True)
    r_on = Retriever(backend, emb, NoOpReranker(), on_cfg,
                     complete=_Comp('{"hypothetical":"假设答案"}'), now_fn=lambda: _NOW)
    on = await r_on.retrieve("原始问题", filt, k=1)
    assert "b" in {h.record.id for h in on}


async def test_complete_none_degrades_to_base(mock_embedder):
    # 开关全开但 complete=None → planner 不启用 → 与基础检索结果完全一致（零行为变更）
    emb = mock_embedder(dimension=64)
    backend = SqliteVecBackend(":memory:", dimension=64)
    for rid, text in [("a", "cat"), ("b", "dog"), ("c", "fish")]:
        v = (await emb.embed([text]))[0]
        backend.upsert([_rec(text, v, rid)])
    filt = MemoryFilter(owner_id="u1", kind="k")
    enh = RetrievalConfig(use_mmr=False, use_entity_recall=True,
                          use_multi_query=True, use_hyde=True)
    base = RetrievalConfig(use_mmr=False)
    got = [h.record.id for h in await _retriever(backend, emb, enh).retrieve("cat", filt, k=5)]
    exp = [h.record.id for h in await _retriever(backend, emb, base).retrieve("cat", filt, k=5)]
    assert got == exp


# ---------- 相关性下限：整条链上唯一的绝对相关性信号 ----------

class _ScoringReranker:
    """按 id 给固定精排分，模拟真实端点。"""
    def __init__(self, scores: dict):
        self._s = scores

    async def rerank(self, query, candidates):
        from harness.memory.reranker import RERANK_SCORE_KEY
        for c in candidates:
            if c.record.id in self._s:
                c.components[RERANK_SCORE_KEY] = self._s[c.record.id]
        return sorted(candidates,
                      key=lambda c: c.components.get(RERANK_SCORE_KEY, 0.0), reverse=True)


async def _two_doc_backend(emb):
    b = SqliteVecBackend(":memory:", dimension=64)
    for rid, t in [("hi", "相关文档"), ("lo", "无关文档")]:
        v = (await emb.embed([t]))[0]
        b.upsert([_rec(t, v, rid)])
    return b


async def test_low_relevance_hits_are_dropped(mock_embedder):
    """回归：查「厨具」在只有 AI 资料的知识库里也能返回满满一屏。

    根因是绝对相似度在链上被销毁两次——RRF 只用排名，_minmax 又在候选集内归一化
    （最好的那条永远得 1.0）。精排分是唯一幸存的绝对信号，早先却被用完即弃。
    """
    emb = mock_embedder(dimension=64)
    b = await _two_doc_backend(emb)
    cfg = RetrievalConfig(use_keyword=False, use_mmr=False, rerank_min_score=0.35)
    hits = await _retriever(b, emb, cfg,
                            _ScoringReranker({"hi": 0.80, "lo": 0.26})).retrieve(
        "查询", MemoryFilter(owner_id="u1"), k=10)
    assert [h.record.id for h in hits] == ["hi"], "低于下限的候选必须丢掉"


async def test_all_below_floor_returns_empty(mock_embedder):
    """全都不够相关 → 返回空，让上层如实说「知识库里没有」，而不是倒一堆无关的出来。"""
    emb = mock_embedder(dimension=64)
    b = await _two_doc_backend(emb)
    cfg = RetrievalConfig(use_keyword=False, use_mmr=False, rerank_min_score=0.35)
    hits = await _retriever(b, emb, cfg,
                            _ScoringReranker({"hi": 0.25, "lo": 0.10})).retrieve(
        "厨具", MemoryFilter(owner_id="u1"), k=10)
    assert hits == []


async def test_floor_off_by_default_keeps_everything(mock_embedder):
    """默认关闭：分数量纲随精排模型而变，不实测就开会误伤。"""
    emb = mock_embedder(dimension=64)
    b = await _two_doc_backend(emb)
    cfg = RetrievalConfig(use_keyword=False, use_mmr=False)   # 未设 rerank_min_score
    hits = await _retriever(b, emb, cfg,
                            _ScoringReranker({"hi": 0.25, "lo": 0.10})).retrieve(
        "查询", MemoryFilter(owner_id="u1"), k=10)
    assert len(hits) == 2


async def test_rerank_failure_does_not_empty_the_library(mock_embedder):
    """精排端点故障时降级为原序、不带分数——此时一条都不能丢。

    否则一次网络抖动就会让整个知识库看起来是空的，用户会以为资料没存进去。
    """
    emb = mock_embedder(dimension=64)
    b = await _two_doc_backend(emb)
    cfg = RetrievalConfig(use_keyword=False, use_mmr=False, rerank_min_score=0.35)
    hits = await _retriever(b, emb, cfg, NoOpReranker()).retrieve(   # 无分数 = 降级
        "查询", MemoryFilter(owner_id="u1"), k=10)
    assert len(hits) == 2, "精排挂掉时必须放行全部，不能整库判空"


async def test_all_dropped_is_logged_not_silent(mock_embedder, caplog):
    """全滤光时必须留一条警告：「查询确实无关」与「换模型后阈值失准」结果一模一样，
    都是知识库看起来空的。不记一笔，后者会静默劣化成「资料没存进去」。"""
    import logging
    emb = mock_embedder(dimension=64)
    b = await _two_doc_backend(emb)
    cfg = RetrievalConfig(use_keyword=False, use_mmr=False, rerank_min_score=0.35)
    with caplog.at_level(logging.WARNING, logger="harness.memory.retriever"):
        hits = await _retriever(b, emb, cfg,
                                _ScoringReranker({"hi": 0.25, "lo": 0.10})).retrieve(
            "厨具", MemoryFilter(owner_id="u1"), k=10)
    assert hits == []
    assert any("相关性下限" in r.getMessage() for r in caplog.records), "全部滤光必须留痕"
