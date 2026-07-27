# tests/test_reranker.py
from types import SimpleNamespace

import httpx
import pytest

from harness.memory.reranker import HttpReranker, NoOpReranker, Reranker


async def test_noop_preserves_order():
    r = NoOpReranker()
    items = ["a", "b", "c"]
    assert await r.rerank("q", items) == ["a", "b", "c"]


def test_protocol_has_rerank():
    assert hasattr(Reranker, "rerank")


# ---- HttpReranker ----

def _cand(text):
    """模拟 retriever 传入的 ScoredHit：有 .record.text。"""
    return SimpleNamespace(record=SimpleNamespace(text=text))


def _texts(candidates):
    return [c.record.text for c in candidates]


class _FakeResp:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


class _FakeClient:
    data: dict | None = None          # 端点返回体
    exc: Exception | None = None      # 注入异常
    calls: list = []                  # 记录请求

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None, headers=None):
        _FakeClient.calls.append({"url": url, "json": json, "headers": headers})
        if _FakeClient.exc is not None:
            raise _FakeClient.exc
        return _FakeResp(_FakeClient.data)


@pytest.fixture
def fake_httpx(monkeypatch):
    _FakeClient.data = None
    _FakeClient.exc = None
    _FakeClient.calls = []
    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)
    return _FakeClient


async def test_reorders_by_relevance_score(fake_httpx):
    fake_httpx.data = {"results": [
        {"index": 2, "relevance_score": 0.9},
        {"index": 0, "relevance_score": 0.5},
        {"index": 1, "relevance_score": 0.1}]}
    r = HttpReranker("https://api.x.cn/v1", "sk-1", "bge-reranker")
    out = await r.rerank("q", [_cand("a"), _cand("b"), _cand("c")])
    assert _texts(out) == ["c", "a", "b"]


async def test_hit_first_then_rest_in_original_order(fake_httpx):
    # 端点只返回部分（如 top_n 截断）：命中在前，未命中按原序接尾，不丢候选
    fake_httpx.data = {"results": [{"index": 1, "relevance_score": 0.9}]}
    r = HttpReranker("https://api.x.cn/v1", "sk-1", "bge-reranker")
    out = await r.rerank("q", [_cand("a"), _cand("b"), _cand("c")])
    assert _texts(out) == ["b", "a", "c"]


async def test_falls_back_to_original_on_error(fake_httpx):
    fake_httpx.exc = httpx.ConnectError("boom")
    r = HttpReranker("https://api.x.cn/v1", "sk-1", "bge-reranker")
    cands = [_cand("a"), _cand("b"), _cand("c")]
    out = await r.rerank("q", cands)
    assert _texts(out) == ["a", "b", "c"]


async def test_empty_results_keeps_order(fake_httpx):
    fake_httpx.data = {"results": []}
    r = HttpReranker("https://api.x.cn/v1", "sk-1", "bge-reranker")
    out = await r.rerank("q", [_cand("a"), _cand("b")])
    assert _texts(out) == ["a", "b"]


async def test_single_candidate_still_scored(fake_httpx):
    # len==1 无法重排序（no-op），但仍须调端点给这唯一候选打分：这条分数在
    # Retriever 下限过滤和 knowledge.search 绝对相关度显示里都要用，不能跳过。
    # 变异证伪：把 rerank 短路退回 `len(candidates) <= 1 or ...`（≤1 直接 return
    # 不打分），本测试的 calls 断言与 components 断言都会失败。
    from harness.memory.reranker import RERANK_SCORE_KEY
    fake_httpx.data = {"results": [{"index": 0, "relevance_score": 0.07}]}
    r = HttpReranker("https://api.x.cn/v1", "sk-1", "bge-reranker")
    cand = _cand("a")
    cand.components = {}
    out = await r.rerank("q", [cand])
    assert _texts(out) == ["a"]                       # 1 条重排序是 no-op
    assert len(fake_httpx.calls) == 1                 # 但确实发起了打分请求
    assert fake_httpx.calls[-1]["json"]["documents"] == ["a"]   # 单文档入参合法
    assert cand.components[RERANK_SCORE_KEY] == 0.07  # 分数写进了 components


async def test_single_candidate_endpoint_failure_no_score(fake_httpx):
    # 端点故障降级语义对单候选同样成立：原序返回、不写分、不崩。
    from harness.memory.reranker import RERANK_SCORE_KEY
    fake_httpx.exc = httpx.ConnectError("boom")
    r = HttpReranker("https://api.x.cn/v1", "sk-1", "bge-reranker")
    cand = _cand("a")
    cand.components = {}
    out = await r.rerank("q", [cand])
    assert _texts(out) == ["a"]
    assert RERANK_SCORE_KEY not in cand.components


async def test_empty_query_skips_endpoint(fake_httpx):
    # 空 query 没东西可打分，短路返回原候选、不发请求、不崩（保留原语义）。
    r = HttpReranker("https://api.x.cn/v1", "sk-1", "bge-reranker")
    out = await r.rerank("   ", [_cand("a"), _cand("b")])
    assert _texts(out) == ["a", "b"]
    assert fake_httpx.calls == []


async def test_empty_candidates_skips_endpoint(fake_httpx):
    # 空候选也无东西可打分，短路返回、不发请求。
    r = HttpReranker("https://api.x.cn/v1", "sk-1", "bge-reranker")
    out = await r.rerank("q", [])
    assert out == []
    assert fake_httpx.calls == []


async def test_request_payload_and_url(fake_httpx):
    fake_httpx.data = {"results": [{"index": 0, "relevance_score": 1.0}]}
    r = HttpReranker("https://api.x.cn/v1/", "sk-1", "bge-reranker", top_n=1)
    await r.rerank("我的问题", [_cand("a"), _cand("b")])
    call = fake_httpx.calls[-1]
    assert call["url"] == "https://api.x.cn/v1/rerank"          # 去重尾斜杠
    assert call["json"]["model"] == "bge-reranker"
    assert call["json"]["query"] == "我的问题"
    assert call["json"]["documents"] == ["a", "b"]
    assert call["json"]["top_n"] == 1
    assert call["headers"]["Authorization"] == "Bearer sk-1"


# ---- DashScope / 千问 qwen rerank 格式 ----

async def test_dashscope_reorders_by_score(fake_httpx):
    # 千问响应：结果在 output.results 里
    fake_httpx.data = {"output": {"results": [
        {"index": 2, "relevance_score": 0.9},
        {"index": 0, "relevance_score": 0.5},
        {"index": 1, "relevance_score": 0.1}]}}
    r = HttpReranker(
        "https://ws-x.cn-beijing.maas.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
        "sk-1", "qwen3-rerank", style="dashscope")
    out = await r.rerank("q", [_cand("a"), _cand("b"), _cand("c")])
    assert _texts(out) == ["c", "a", "b"]


async def test_dashscope_request_shape(fake_httpx):
    fake_httpx.data = {"output": {"results": [{"index": 0, "relevance_score": 1.0}]}}
    url = "https://ws-x.cn-beijing.maas.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    r = HttpReranker(url, "sk-1", "qwen3-rerank", style="dashscope", top_n=2)
    await r.rerank("什么是文本排序模型", [_cand("a"), _cand("b"), _cand("c")])
    call = fake_httpx.calls[-1]
    assert call["url"] == url                                    # 完整 endpoint，不追加 /rerank
    assert call["json"]["model"] == "qwen3-rerank"
    assert call["json"]["input"] == {
        "query": "什么是文本排序模型", "documents": ["a", "b", "c"]}
    assert call["json"]["parameters"]["top_n"] == 2              # min(top_n, 候选数)
    assert call["json"]["parameters"]["return_documents"] is False
    assert call["headers"]["Authorization"] == "Bearer sk-1"


async def test_dashscope_falls_back_on_error(fake_httpx):
    fake_httpx.exc = httpx.ConnectError("boom")
    r = HttpReranker("https://ws-x.maas.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
                     "sk-1", "qwen3-rerank", style="dashscope")
    out = await r.rerank("q", [_cand("a"), _cand("b"), _cand("c")])
    assert _texts(out) == ["a", "b", "c"]


def test_emit_rerank_usage_dashscope_reports_per_model():
    from harness.progress import set_emitter, reset_emitter
    from harness.events import ModelUsage
    from harness.memory.reranker import _emit_rerank_usage
    seen = []
    tok = set_emitter(seen.append)
    try:
        _emit_rerank_usage({"usage": {"total_tokens": 50}}, "rr-model")
    finally:
        reset_emitter(tok)
    mu = [e for e in seen if isinstance(e, ModelUsage)]
    assert len(mu) == 1 and mu[0].model == "rr-model" and mu[0].usage.total_tokens == 50


def test_emit_rerank_usage_missing_is_noop():
    from harness.progress import set_emitter, reset_emitter
    from harness.memory.reranker import _emit_rerank_usage
    seen = []
    tok = set_emitter(seen.append)
    try:
        _emit_rerank_usage({"results": []}, "m")   # 端点未返回 usage
    finally:
        reset_emitter(tok)
    assert seen == []


async def test_reorder_records_scores_into_components():
    """真实精排必须把分数写进 components——否则 Retriever 的相关性下限永远不触发。

    这条补的是桩测试盖不住的缺口：用 _ScoringReranker 之类的桩测下限，只证明了
    「有分数时会过滤」，没证明「真实链路上会有分数」。
    """
    from harness.memory.reranker import RERANK_SCORE_KEY, _reorder

    class _C:
        def __init__(self, name):
            self.name = name
            self.components = {}

    a, b = _C("a"), _C("b")
    out = _reorder([a, b], [{"index": 0, "relevance_score": 0.2},
                            {"index": 1, "relevance_score": 0.9}])
    assert [c.name for c in out] == ["b", "a"]           # 仍按分数重排
    assert a.components[RERANK_SCORE_KEY] == 0.2         # 分数留存，不再用完即弃
    assert b.components[RERANK_SCORE_KEY] == 0.9
