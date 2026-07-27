# src/harness/memory/query_planner.py
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass

from ..llm.openai_compat import json_output

log = logging.getLogger(__name__)


@dataclass
class QueryPlan:
    """一次查询规划的产物；未启用的路对应字段为空。"""
    variants: list[str]      # 多查询改写
    hypothetical: str        # HyDE 假设文档
    entity_keys: list[str]   # 抽取的点分实体键


EMPTY_PLAN = QueryPlan(variants=[], hypothetical="", entity_keys=[])


def _strip_fence(raw: str) -> str:
    s = (raw or "").strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else ""
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    return s.strip()


class QueryPlanner:
    """查询期一次 LLM 调用，按启用的开关产出多查询改写 / HyDE 假设文档 / 实体键，
    用于扩大召回。未启用（无 completer 或三路全关）或调用失败/超时/解析失败时返回
    EMPTY_PLAN，检索据此降级回基础召回。检索在聊天首字关键路径上，故必须带超时。"""

    def __init__(self, complete, *, multi_query: bool = False, multi_query_n: int = 3,
                 hyde: bool = False, entity: bool = False, timeout_s: float = 2.0) -> None:
        self._complete = complete
        self._multi_query = multi_query
        self._multi_query_n = max(1, multi_query_n)
        self._hyde = hyde
        self._entity = entity
        self._timeout_s = timeout_s

    @property
    def enabled(self) -> bool:
        return self._complete is not None and (
            self._multi_query or self._hyde or self._entity)

    def _build_prompt(self) -> str:
        fields: list[str] = []
        if self._multi_query:
            fields.append(
                f"\"variants\": {self._multi_query_n} 条与原查询等价、换一种说法的查询（字符串数组）")
        if self._hyde:
            fields.append(
                "\"hypothetical\": 一段简短的假设答案文本（就当你已知道答案，直接写出来）")
        if self._entity:
            fields.append(
                "\"entity_keys\": 查询涉及的点分实体键数组，如 [\"user.pref.language\"]；无则空数组")
        return ("你是检索查询规划器。针对用户的查询，产出用于扩大召回的辅助信息。"
                "只输出一个 JSON 对象，包含字段：" + "；".join(fields) + "。"
                "不要输出 JSON 以外的任何文字。")

    async def plan(self, query_text: str) -> QueryPlan:
        if not self.enabled:
            return EMPTY_PLAN
        try:
            with json_output():
                raw = await asyncio.wait_for(
                    self._complete(self._build_prompt(), query_text), self._timeout_s)
        except Exception as e:   # 含 asyncio.TimeoutError：超时/异常一律降级
            log.warning("query plan failed/timeout: %s", e)
            return EMPTY_PLAN
        return self._parse(raw)

    def _parse(self, raw: str) -> QueryPlan:
        try:
            data = json.loads(_strip_fence(raw))
        except (json.JSONDecodeError, TypeError):
            return EMPTY_PLAN
        if not isinstance(data, dict):
            return EMPTY_PLAN
        variants: list[str] = []
        if self._multi_query and isinstance(data.get("variants"), list):
            variants = [str(v) for v in data["variants"] if str(v).strip()][:self._multi_query_n]
        hypothetical = ""
        if self._hyde and isinstance(data.get("hypothetical"), str):
            hypothetical = data["hypothetical"].strip()
        entity_keys: list[str] = []
        if self._entity and isinstance(data.get("entity_keys"), list):
            entity_keys = [str(k).strip() for k in data["entity_keys"] if str(k).strip()]
        return QueryPlan(variants=variants, hypothetical=hypothetical, entity_keys=entity_keys)
