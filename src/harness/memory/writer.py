# src/harness/memory/writer.py
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

from ..llm.openai_compat import json_output
from .record import MemType, MemoryFilter, MemoryRecord

log = logging.getLogger(__name__)

_EXTRACT_SYS = (
    "你是记忆提炼器。从下面的文本中提炼**值得长期记住**的事实——"
    "用户偏好、目标、稳定属性、学到的方法或结论；忽略寒暄、一次性内容、临时上下文。"
    "输出一个 JSON 对象 {\"items\":[...]}，items 为数组，每个元素：{\"text\": 简洁事实, "
    "\"mem_type\": \"semantic\"|\"episodic\"|\"procedural\", \"entity_key\": 点分实体键如 "
    "\"user.pref.language\"（无明确实体则空串）, \"importance\": 0~1 的浮点}。"
    "importance 按此量表评分："
    "0.9~1.0=用户身份/长期偏好/核心目标；"
    "0.6~0.8=稳定的方法、结论或重要约束；"
    "0.4~0.5=一般事实；"
    "0.1~0.3=次要细节或时效性强、易过期的内容。"
    "没有值得记的内容就令 items 为 []。只输出 JSON，不要解释。"
)


@dataclass
class ExtractedFact:
    text: str
    mem_type: MemType
    entity_key: str = ""
    importance: float = 0.5


@dataclass
class MemoryOp:
    op: str                              # "ADD" | "NOOP" | "REPLACE"
    fact: ExtractedFact | None           # 由 fact_index 解析而来；NOOP 也带（应用时跳过）
    supersede_ids: list[str] = field(default_factory=list)


def _strip_fence(raw: str) -> str:
    s = (raw or "").strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else ""
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    return s.strip()


def _parse_facts(raw: str) -> list[ExtractedFact]:
    # 解析失败即「本轮一条记忆都不写」，且调用方看不出区别（写入本就 best-effort）——
    # 必须出声，否则智能写入整个停工都无人知晓。日志只记数量：raw 是用户事实，不入日志。
    try:
        data = json.loads(_strip_fence(raw))
    except (json.JSONDecodeError, TypeError):
        log.warning("记忆提炼：模型输出不是合法 JSON（%d 字），本轮不写入任何记忆",
                    len(raw or ""))
        return []
    if isinstance(data, dict):        # json_object 信封 {"items":[...]}；模型漏包时下面兜底裸数组
        data = data.get("items")
    if not isinstance(data, list):
        log.warning("记忆提炼：模型输出是 %s、不是数组，本轮不写入任何记忆",
                    type(data).__name__)
        return []
    facts: list[ExtractedFact] = []
    for item in data:
        if not isinstance(item, dict) or not item.get("text"):
            continue
        try:
            mem_type = MemType(item.get("mem_type", "semantic"))
        except ValueError:
            mem_type = MemType.SEMANTIC
        try:
            importance = float(item.get("importance", 0.5))
        except (TypeError, ValueError):
            importance = 0.5
        facts.append(ExtractedFact(
            text=str(item["text"]), mem_type=mem_type,
            entity_key=str(item.get("entity_key") or ""), importance=importance))
    return facts


_RECONCILE_SYS = (
    "你在维护一个记忆库。给你【新事实】列表和与之相关的【已有记忆】。"
    "对每个新事实，决定操作并输出一个 JSON 对象 {\"items\":[...]}，items 为数组，元素："
    "{\"op\": \"ADD\"|\"NOOP\"|\"REPLACE\", \"fact_index\": 新事实下标, "
    "\"supersede_ids\": 该事实要作废的已有记忆 id 列表（仅 REPLACE）}。"
    "规则：与某条已有记忆语义等同 → NOOP；是同一实体的更新、或与某条已有记忆矛盾 → "
    "REPLACE 并在 supersede_ids 填被取代的已有记忆 id；全新信息 → ADD。只输出 JSON。"
)


def _parse_ops(raw: str, facts: list[ExtractedFact]) -> list[MemoryOp]:
    # 「退化为全部 ADD」= 调和这步等于没跑：不去重、不消矛盾，记忆库会慢慢灌满重复与自相
    # 矛盾的条目，而全程无声。必须出声。日志只记数量：raw 是用户事实，不入日志。
    try:
        data = json.loads(_strip_fence(raw))
        if isinstance(data, dict):    # json_object 信封 {"items":[...]}；模型漏包时兜底裸数组
            data = data.get("items")
        if not isinstance(data, list):
            raise ValueError("not a list")
    except (json.JSONDecodeError, TypeError, ValueError):
        log.warning("记忆调和：模型输出不是 JSON 数组（%d 字），退化为全部 ADD"
                    "——本轮不去重、不消矛盾", len(raw or ""))
        return [MemoryOp("ADD", f) for f in facts]
    ops: list[MemoryOp] = []
    covered: set[int] = set()
    bad = 0
    for item in data:
        if not isinstance(item, dict):
            bad += 1
            continue
        op = item.get("op")
        idx = item.get("fact_index")
        if op not in ("ADD", "NOOP", "REPLACE"):
            bad += 1
            continue
        if not isinstance(idx, int) or not (0 <= idx < len(facts)):
            bad += 1
            continue
        covered.add(idx)
        sup = item.get("supersede_ids", []) if op == "REPLACE" else []
        if not isinstance(sup, list):
            sup = []
        ops.append(MemoryOp(op, facts[idx], [str(x) for x in sup]))
    uncovered = [i for i in range(len(facts)) if i not in covered]
    for i in uncovered:
        ops.append(MemoryOp("ADD", facts[i]))
    if bad or uncovered:      # 部分退化：这些事实没经调和就直接进库，同样是悄悄漏判
        log.warning("记忆调和：%d 项判定无效被跳过，%d/%d 条事实没拿到判定、按 ADD 处理",
                    bad, len(uncovered), len(facts))
    return ops


class MemoryWriter:
    """LLM 驱动的智能写入：提炼 → 找候选 → 调和 → 应用。"""

    def __init__(self, backend, embedder, retriever, complete, *,
                 extract_complete=None, candidate_k: int = 5,
                 ttl_by_type: dict | None = None, now_fn=None) -> None:
        """complete 用于调和（_reconcile），extract_complete 用于提炼（_extract）。

        两步吃的能力不同，故可分开配：提炼是机械活，换便宜小模型无妨；调和是判断题且
        后果不可逆——判 REPLACE 会 set_superseded 永久作废旧记忆，判错不是省钱是毁数据，
        默认就该用主模型。extract_complete 省略时二者同源，行为与旧版一致。
        """
        self._backend = backend
        self._embedder = embedder
        self._retriever = retriever
        self._complete = complete
        self._extract_complete = extract_complete or complete
        self._candidate_k = candidate_k
        self._ttl_by_type = ttl_by_type
        import time
        self._now_fn = now_fn or (lambda: int(time.time()))

    async def _extract(self, text: str) -> list[ExtractedFact]:
        try:
            with json_output():
                raw = await self._extract_complete(_EXTRACT_SYS, text)
        except Exception as e:
            log.warning("memory extract LLM failed: %s", e)
            return []
        return _parse_facts(raw)

    async def _gather_candidates(self, owner_id: str, kind: str,
                                 facts: list[ExtractedFact]) -> list[MemoryRecord]:
        cand: dict[str, MemoryRecord] = {}
        for f in facts:
            if f.entity_key:
                for r in self._backend.list_by_entity(owner_id, kind, f.entity_key):
                    cand[r.id] = r
            hits = await self._retriever.retrieve(
                f.text, MemoryFilter(owner_id=owner_id, kind=kind), self._candidate_k)
            for h in hits:
                cand[h.record.id] = h.record
        return list(cand.values())

    async def _reconcile(self, facts: list[ExtractedFact],
                         candidates: list[MemoryRecord]) -> list[MemoryOp]:
        if not candidates:
            return [MemoryOp("ADD", f) for f in facts]
        payload = json.dumps({
            "new_facts": [{"fact_index": i, "text": f.text} for i, f in enumerate(facts)],
            "existing_memories": [{"id": c.id, "text": c.text, "entity_key": c.entity_key}
                                  for c in candidates],
        }, ensure_ascii=False)
        try:
            with json_output():
                raw = await self._complete(_RECONCILE_SYS, payload)
        except Exception as e:
            log.warning("memory reconcile LLM failed: %s", e)
            return [MemoryOp("ADD", f) for f in facts]
        return _parse_ops(raw, facts)

    async def _apply(self, owner_id: str, kind: str,
                     ops: list[MemoryOp]) -> list[str]:
        new_ids: list[str] = []
        for op in ops:
            if op.op == "NOOP" or op.fact is None:
                continue
            try:
                version = 1
                if op.op == "REPLACE" and op.supersede_ids:
                    old = self._backend.get(op.supersede_ids)
                    version = max([r.version for r in old], default=0) + 1
                    self._backend.set_superseded(op.supersede_ids)
                expires_at = 0
                if self._ttl_by_type:
                    ttl = self._ttl_by_type.get(op.fact.mem_type.value, 0)
                    if ttl > 0:
                        expires_at = self._now_fn() + ttl
                vec = (await self._embedder.embed([op.fact.text]))[0]
                rec = MemoryRecord(
                    owner_id=owner_id, kind=kind, mem_type=op.fact.mem_type,
                    text=op.fact.text, embedding=vec, entity_key=op.fact.entity_key,
                    importance=op.fact.importance, version=version, source="extract",
                    expires_at=expires_at)
                new_ids.extend(self._backend.upsert([rec]))
            except Exception as e:                       # 单个 op 失败不影响其余（best-effort）
                log.warning("memory apply op failed (%s): %s", op.op, e)
                continue
        return new_ids

    async def write(self, owner_id: str, kind: str, text: str) -> list[str]:
        """智能写入：提炼→找候选→调和→应用，返回新写入记录 id。best-effort，异常安全。"""
        if not text or not text.strip():
            return []
        # 分阶段计时：本方法被 chat 的 gen() await，跑完这轮 run 才算结束、SSE 流才关闭、
        # 前端才解除 busy —— 也就是说它不拖慢答案（RunFinished 早已送出），却拖着用户
        # 不能发下一句。要不要把它改成 fire-and-forget（代价：进程重启会丢在写的记忆），
        # 得看真实耗时；而分到阶段才知道该改哪儿——若大头在调和，先关它的思考链可能就够了。
        _t = time.monotonic()
        facts = await self._extract(text)
        ms_ext = round((time.monotonic() - _t) * 1000)
        if not facts:
            return []
        _t = time.monotonic()
        try:
            candidates = await self._gather_candidates(owner_id, kind, facts)
        except Exception as e:                           # 找候选失败：降级为无候选（全 ADD）
            log.warning("memory gather candidates failed: %s", e)
            candidates = []
        ms_cand = round((time.monotonic() - _t) * 1000)
        _t = time.monotonic()
        ops = await self._reconcile(facts, candidates)
        ms_rec = round((time.monotonic() - _t) * 1000)
        _t = time.monotonic()
        new_ids = await self._apply(owner_id, kind, ops)
        ms_apply = round((time.monotonic() - _t) * 1000)
        # 判定分布是「调和到底在不在工作」的正面信号：有候选却清一色 ADD、NOOP/REPLACE 恒为 0，
        # 就说明它在空转（模型没判、或判定被静默丢弃），光靠失败告警看不出这种情况。
        _dist = {k: sum(1 for o in ops if o.op == k) for k in ("ADD", "NOOP", "REPLACE")}
        log.info("记忆写入 提炼=%d条 候选=%d 判定=%s 新增=%d 耗时=%dms"
                 "（提炼%d/候选%d/调和%d/落库%d）",
                 len(facts), len(candidates), _dist, len(new_ids),
                 ms_ext + ms_cand + ms_rec + ms_apply, ms_ext, ms_cand, ms_rec, ms_apply)
        return new_ids
