# src/harness/skills/matcher.py
"""SkillMatcher：按触发关键词把用户消息确定性地匹配到一个技能。

用于路由层「主动挂载」——解决模型几乎从不主动 load_skill（实测 0/440）的问题：
命中的技能剧本由编排器注入 planner（拆解蓝本）与简单直答（参考），而非等模型自觉加载。
纯关键词子串匹配，零 LLM 成本、确定可预期。
"""
from __future__ import annotations

from .registry import Skill, SkillRegistry


class SkillMatcher:
    def __init__(self, registry: SkillRegistry) -> None:
        self._registry = registry

    def match(self, message: str) -> Skill | None:
        """返回触发词在 message 中命中最多的技能；无命中或空消息返回 None。

        平局取技能名字典序靠前者（registry.names 有序），保证结果稳定可预期。
        """
        if not message or not message.strip():
            return None
        text = message.lower()
        best: Skill | None = None
        best_hits = 0
        for name in self._registry.names():
            s = self._registry.get(name)
            if s is None or not s.triggers:
                continue
            hits = sum(1 for t in s.triggers if t and t.lower() in text)
            if hits > best_hits:
                best_hits = hits
                best = s
        return best
