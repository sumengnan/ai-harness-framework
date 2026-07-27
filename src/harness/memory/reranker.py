# src/harness/memory/reranker.py
from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import httpx

logger = logging.getLogger("harness.memory.reranker")


@runtime_checkable
class Reranker(Protocol):
    """精排可插拔切点。默认 NoOpReranker；日后接 LLM/cross-encoder 实现同签名。"""

    async def rerank(self, query: str, candidates: list) -> list:
        ...


class NoOpReranker:
    """不改序，原样返回候选。"""

    async def rerank(self, query: str, candidates: list) -> list:
        return candidates


class HttpReranker:
    """调用 rerank REST 端点做精排，按 style 适配不同厂商的请求/响应格式。

    style="openai"（默认，兼容 Cohere/Jina/SiliconFlow）：
      请求 POST {base}/rerank  {model, query, documents, [top_n]}
      响应 {"results": [{"index", "relevance_score"}, ...]}
    style="dashscope"（阿里云百炼 / 千问 qwen3-rerank）：
      请求 POST {base}（base 即完整 endpoint，不追加 /rerank）
           {model, input:{query, documents}, parameters:{return_documents, [top_n]}}
      响应 {"output": {"results": [{"index", "relevance_score"}, ...]}}

    失败哲学：网络/超时/HTTP/格式异常一律降级为「原序返回」，绝不打断检索或聊天
    （与 L3 检索「失败不影响本轮」一致）。top_n 截断时命中的排前、未命中的按原序接尾，
    不丢候选，最终 top-k 截断仍交给 Retriever。
    """

    def __init__(self, base_url: str, api_key: str, model: str, *,
                 style: str = "openai", timeout: float = 30.0,
                 top_n: int | None = None) -> None:
        self._style = style
        self._base = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._top_n = top_n

    def _build_request(self, query: str, documents: list[str]) -> tuple[str, dict]:
        top_n = min(self._top_n, len(documents)) if self._top_n else None
        if self._style == "dashscope":
            params: dict = {"return_documents": False}
            if top_n:
                params["top_n"] = top_n
            return self._base, {                       # base 即完整 endpoint
                "model": self._model,
                "input": {"query": query, "documents": documents},
                "parameters": params}
        payload: dict = {"model": self._model, "query": query, "documents": documents}
        if top_n:
            payload["top_n"] = top_n
        return self._base + "/rerank", payload

    def _parse_results(self, data: dict) -> list:
        if self._style == "dashscope":
            return (data.get("output") or {}).get("results", [])
        return data.get("results", [])

    async def rerank(self, query: str, candidates: list) -> list:
        # 空候选或空 query 没东西可打分，短路返回。注意 len==1 不在此列：单条虽无法
        # 重排序（那是 no-op），但它的精排分在 Retriever 下限过滤和 knowledge.search
        # 绝对相关度显示里都要用，必须照常调端点打分，不能跳过。
        if not candidates or not query or not query.strip():
            return candidates
        documents = [_doc_text(c) for c in candidates]
        url, payload = self._build_request(query, documents)
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                body = resp.json()
                results = self._parse_results(body)
        except Exception as e:  # 网络/超时/HTTP/格式错误一律降级保序
            logger.warning("rerank 端点调用失败，退回原序：%s", e)
            return candidates
        _emit_rerank_usage(body, self._model)   # 端点返回 usage 时按模型上报（尽力而为）
        return _reorder(candidates, results)


def _emit_rerank_usage(body: dict, model: str) -> None:
    """从 rerank 响应里取 usage 上报（DashScope 在 body.usage.total_tokens；OpenAI 兼容在 body.usage）。
    端点不返回 usage 时静默跳过；emitter 未设时 no-op。绝不影响 rerank 本职。"""
    try:
        usage = (body or {}).get("usage") or ((body or {}).get("output") or {}).get("usage")
        if not usage:
            return
        total = int(usage.get("total_tokens") or usage.get("prompt_tokens") or 0)
        if total <= 0:
            return
        from harness.events import ModelUsage
        from harness.progress import emit
        from harness.usage import Usage, effective_cost
        u = Usage(prompt_tokens=total, completion_tokens=0, total_tokens=total)
        emit(ModelUsage(usage=u, cost_usd=effective_cost(u, model, {}),
                        attempts=1, latency_ms=0.0, model=model))
    except Exception:
        pass


def _doc_text(candidate) -> str:
    """取候选正文用于打分。兼容 ScoredHit（.record.text）与裸字符串。"""
    rec = getattr(candidate, "record", None)
    if rec is not None:
        return getattr(rec, "text", "") or ""
    return candidate if isinstance(candidate, str) else str(candidate)


RERANK_SCORE_KEY = "rerank"


def _reorder(candidates: list, results: list) -> list:
    """按 results 的 relevance_score 降序重排：命中的排前，未命中的按原序接尾（不丢候选）。

    顺带把分数记进 candidate.components[RERANK_SCORE_KEY]。这个分是整条检索链上**唯一**
    的绝对相关性信号：RRF 只用排名、_minmax 又在候选集内归一化（最好的那条永远得 1.0），
    绝对相似度到打分阶段已经荡然无存。早先这里把 relevance_score 用完即弃，于是
    「查厨具」在一个只有 AI 资料的知识库里也能返回满满一屏——没有任何一处能说出
    「都不够相关」。留下它，供 Retriever 按下限过滤（见 RetrievalConfig.rerank_min_score）。
    """
    scored: list[tuple[int, float]] = []
    for r in results:
        idx = r.get("index")
        if isinstance(idx, int) and 0 <= idx < len(candidates):
            scored.append((idx, r.get("relevance_score", 0.0)))
    if not scored:                        # 端点未给可用结果 → 原序
        return candidates
    for idx, s in scored:
        comp = getattr(candidates[idx], "components", None)
        if isinstance(comp, dict):
            comp[RERANK_SCORE_KEY] = s
    scored.sort(key=lambda t: t[1], reverse=True)
    seen = {idx for idx, _ in scored}
    ordered = [candidates[idx] for idx, _ in scored]
    ordered += [c for i, c in enumerate(candidates) if i not in seen]
    return ordered
