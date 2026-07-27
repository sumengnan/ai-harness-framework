# src/harness/tools/builtins/episode_tools.py
from __future__ import annotations

from pydantic import BaseModel

from ..base import Tool
from ...memory.episodic import EpisodicMemory


class RecallEpisodesTool(Tool):
    name = "recall_episodes"
    description = "检索过往相似任务的经验（做法与成败），参考它来完成当前任务。"

    class Params(BaseModel):
        query: str
        k: int | None = None

    def __init__(self, episodic: EpisodicMemory, default_k: int = 3) -> None:
        self._episodic = episodic
        self._default_k = default_k

    async def run(self, params: "RecallEpisodesTool.Params") -> str:
        k = params.k if params.k is not None else self._default_k
        hits = await self._episodic.recall(params.query, k)
        if not hits:
            return "（无相关历史经验）"
        return "\n\n".join(f"[{i}] {h.text}" for i, h in enumerate(hits, 1))


class RecordEpisodeTool(Tool):
    name = "record_episode"
    description = "把一次任务的经验（做法/教训与成败）记录下来供以后参考。"

    class Params(BaseModel):
        task: str
        lesson: str
        success: bool = True

    def __init__(self, episodic: EpisodicMemory) -> None:
        self._episodic = episodic

    async def run(self, params: "RecordEpisodeTool.Params") -> str:
        await self._episodic.record(params.task, params.lesson, params.success)
        return "已记录经验。"
