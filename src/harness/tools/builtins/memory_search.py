from __future__ import annotations

from pydantic import BaseModel

from ..base import Tool
from ...memory.memory import Memory


class _CollectionSearchTool(Tool):
    """向量库按 collection 检索的公共实现。

    子类只定义「查哪个 scope、对模型怎么自称」：name / description / _default_collection
    / _empty。检索逻辑本身两者完全一致，差异全在语义，所以刻意不把 collection 做成模型
    可传的参数——scope 由构造方按用户注入，模型不该也不能跨用户检索。
    """

    # 子类必须各自定义：空命中文案会被 app/sources.py 的 builder 用前缀判空、被
    # validating/verify 当哨兵匹配，给个兜底默认值只会让漏定义的子类静默记出假来源。
    _default_collection: str
    _empty: str
    # 检索条数的上下限。上限防止把上下文撑爆；下限由子类按用途收紧。
    _K_MIN = 1
    _K_MAX = 50

    class Params(BaseModel):
        query: str
        k: int | None = None

    def __init__(
        self, memory: Memory, collection: str | None = None, default_k: int = 5
    ) -> None:
        self._memory = memory
        self._collection = collection or self._default_collection
        self._default_k = default_k

    async def run(self, params: "_CollectionSearchTool.Params") -> str:
        # 夹到 [_K_MIN, _K_MAX]：模型常自作主张传很小的 k（实测传 3），几条片段根本
        # 覆盖不住知识库里的相关内容，回答就变成「资料里没提到」。光调默认值没用——
        # 显式传参会盖掉默认值，所以要有下限兜底。
        k = params.k if params.k is not None else self._default_k
        k = max(self._K_MIN, min(int(k), self._K_MAX))
        hits = await self._memory.search(params.query, self._collection, k)
        if not hits:
            return self._empty
        lines = []
        for i, h in enumerate(hits, 1):
            src = h.metadata.get("source")
            tag = f"（来源：{src}）" if src else ""
            lines.append(f"[{i}]{tag} {h.text}")
        return "\n".join(lines)


# 知识库空命中的哨兵文案。**这是跨层契约**：app 的每步校验（relevance_check）与交付门的
# grounding 判据都要认出「这次检索什么也没查到」，而它们唯一的依据就是这串文字。
# 从这里 import，别在各处重抄一遍——重抄的后果是改了内核这句话，app 侧会**静默**失效：
# 相关性校验永远判通过、grounding 把空结果当成有依据，没有任何报错提示你。
NO_KNOWLEDGE_HIT = "（未在知识库中检索到相关内容）"


class SearchKnowledgeTool(_CollectionSearchTool):
    """检索用户知识库（上传/保存的资料）。作答的可引用依据，交付门的 grounding 判据。"""

    name = "search_knowledge"
    description = (
        "在用户的知识库中检索相关资料并返回原文片段。知识库装的是用户上传或保存的文档，"
        "是回答学习问题时可引用的依据。"
        "k 常用 10-50（不传则用默认值）：检索片段太少容易漏掉相关资料，"
        "把本来有依据的问题答成「资料里没有」。"
        "注意：这里查不到你自己记下的偏好/结论，那些请用 search_memory。")
    # 知识库检索的下限：这是作答依据的来源，宁可多给几段也别漏。
    _K_MIN = 10
    _default_collection = "knowledge"
    _empty = NO_KNOWLEDGE_HIT


class SearchMemoryTool(_CollectionSearchTool):
    """检索 AI 自己的长期记忆（remember 写入）。与知识库不同，这不是可引用的资料来源。"""

    name = "search_memory"
    description = (
        "检索你自己记下的长期记忆——那些你用 remember 存过的、关于这位用户的偏好、"
        "习惯和过往结论，跨会话有效。"
        "注意：这里没有用户的资料文档，查资料请用 search_knowledge。")
    _default_collection = "memory"
    _empty = "（未检索到相关的长期记忆）"
