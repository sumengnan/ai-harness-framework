# src/harness/memory/episodic.py
from __future__ import annotations

from dataclasses import dataclass

from ..events import RunError, RunFinished
from .memory import Memory


@dataclass
class Episode:
    task: str
    outcome: str
    success: bool

    def to_text(self) -> str:
        status = "成功" if self.success else "失败"
        return f"任务：{self.task}\n结果（{status}）：{self.outcome}"


class EpisodicMemory:
    """绑定 episodes collection 的 Memory 薄门面。复用 ③a 全部向量设施。"""

    def __init__(self, memory: Memory, collection: str = "episodes") -> None:
        self._memory = memory
        self._collection = collection

    async def record(self, task: str, outcome: str, success: bool) -> list[str]:
        # 局限：超长 outcome/task 会被 Memory.add_texts 按 chunk_size 分块存储，
        # recall 可能只召回其中一个碎片。v1 假定 episode 简短；更完整的方案是先用
        # LLM 摘要成短经验再入库（规格列为 OUT of scope）。
        ep = Episode(task, outcome, success)
        # metadata 预留给未来"按 success 过滤召回"（如只看失败案例复盘）；
        # 当前 RecallEpisodesTool 尚未读取 metadata，仅纯语义检索。
        return await self._memory.add_texts(
            [ep.to_text()], self._collection, {"success": success, "task": task[:200]})

    async def recall(self, query: str, k: int):
        return await self._memory.search(query, self._collection, k)


class EpisodeRecorder:
    """事件流包装器：透传事件，run 终止时自动记一条 episode。loop 零改动。"""

    def __init__(self, episodic: EpisodicMemory) -> None:
        self._episodic = episodic

    async def wrap(self, events, task: str):
        outcome, success, terminal = "", False, False
        try:
            async for ev in events:
                if isinstance(ev, RunFinished):
                    # 注意：RunFinished 只代表 run 正常结束，不等于内容真的成功——
                    # 模型自称"做不到"也会被记成 success=True。真正判成败需未来的
                    # LLM-judge（规格列为 OUT of scope）。
                    outcome, success, terminal = ev.message.content or "", True, True
                elif isinstance(ev, RunError):
                    outcome, success, terminal = ev.error, False, True
                yield ev
        finally:
            # 放进 finally：消费者在收到终止事件后 break（生成器被 aclose/GC finalize，
            # GeneratorExit 在 yield 处抛出）时，落库逻辑仍会执行，不丢记录。
            if terminal:   # 只记录跑完（有终止事件）的 run；无终止事件 terminal=False 不落库
                await self._episodic.record(task, outcome, success)
