from __future__ import annotations

from typing import Protocol, runtime_checkable

from openai import AsyncOpenAI


@runtime_checkable
class EmbeddingClient(Protocol):
    dimension: int

    async def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class OpenAICompatibleEmbeddingClient:
    """调用 OpenAI 兼容 /embeddings 端点。端点与聊天端点独立配置。"""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        dimension: int,
        timeout: float = 60.0,
        batch_size: int = 20,
    ) -> None:
        self.dimension = dimension
        self._model = model
        self._batch = max(1, int(batch_size or 20))
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        # 分批：多数厂商对单次条数有上限（DashScope text-embedding-v4 为 20，超了直接
        # 400 InvalidParameter「batch size is invalid」）。而知识库入库是按整篇文档的全部
        # 分块一次性调用的——稍长一点的文档必然超限，表现为「上传失败」。
        # 按上限切片顺序请求，结果按原顺序拼回：分块与向量必须一一对应，错位比失败更糟。
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch):
            out += await self._embed_batch(texts[i:i + self._batch])
        return out

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        # 显式请求目标维度：text-embedding-v4 等模型默认维度可能不是配置值
        # （如默认 1024），不指定就会与向量表 float[dimension] 不符。
        resp = await self._client.embeddings.create(
            model=self._model, input=texts, dimensions=self.dimension)
        # 用量按模型上报：端点返回 usage 时经进度旁路发一条 ModelUsage（model=embedding_model），
        # 让 embedding 也进本轮合计与分模型统计。emitter 未设（如上下文组装、后台记忆写入不在本轮
        # 用量上下文内）时是 no-op，故只捕获「loop 内工具检索」触发的 embedding，是尽力而为。
        _emit_embedding_usage(getattr(resp, "usage", None), self._model)
        # 第三方兼容端点未必保证顺序，按 index 归位
        return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]


def _emit_embedding_usage(usage, model: str) -> None:
    if usage is None:
        return
    try:
        from harness.events import ModelUsage
        from harness.progress import emit
        from harness.usage import Usage, effective_cost
        prompt = int(getattr(usage, "prompt_tokens", 0) or getattr(usage, "total_tokens", 0) or 0)
        total = int(getattr(usage, "total_tokens", 0) or prompt)
        u = Usage(prompt_tokens=prompt, completion_tokens=0, total_tokens=total)
        emit(ModelUsage(usage=u, cost_usd=effective_cost(u, model, {}),
                        attempts=1, latency_ms=0.0, model=model))
    except Exception:   # 用量上报绝不能影响 embedding 本职
        pass
