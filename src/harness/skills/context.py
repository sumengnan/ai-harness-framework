from __future__ import annotations

from ..state import RunState
from ..types import Message, Role
from .registry import SkillRegistry


def resolve_loaded_skills(state: RunState, registry: SkillRegistry) -> list[str]:
    """从消息历史里的 load_skill/unload_skill 工具调用，推导当前加载的技能（有序）。

    纯函数，加载状态的唯一真相是消息历史本身——无共享可变态，天然并发安全、
    随 run 结束自动重置。load 入序去重、unload 移除、registry 中不存在的忽略。
    """
    loaded: list[str] = []
    for m in state.messages:
        for tc in m.tool_calls:
            nm = (tc.arguments or {}).get("name")
            if not isinstance(nm, str):
                continue
            if tc.name == "load_skill":
                if registry.has(nm) and nm not in loaded:
                    loaded.append(nm)
            elif tc.name == "unload_skill":
                if nm in loaded:
                    loaded.remove(nm)
    return loaded


class SkillContextManager:
    """装饰任意 inner ContextManager，注入技能元数据索引与已加载技能正文。

    - 元数据索引常驻，已加载技能正文按需注入，两者都折叠进**单个** system 消息，
      避免多 system 消息的 provider 兼容性问题。
    - build() 是纯函数：加载状态由 state 推导，不改 inner、不改 AgentLoop/RunState。
    - registry 为空时完全透明，等价于直接用 inner。
    """

    def __init__(self, inner, registry: SkillRegistry) -> None:
        self._inner = inner
        self._reg = registry

    def build(self, state: RunState) -> list[Message]:
        msgs = self._inner.build(state)
        index = self._reg.index_text()
        if not index:                      # 无技能：透明透传
            return msgs

        extra = "\n\n" + index
        loaded = resolve_loaded_skills(state, self._reg)
        if loaded:
            bodies = "\n\n".join(f"## 技能：{n}\n{self._reg.body(n)}" for n in loaded)
            extra += "\n\n# 已加载技能详情\n" + bodies

        if msgs and msgs[0].role == Role.SYSTEM:
            merged = Message(role=Role.SYSTEM, content=(msgs[0].content or "") + extra)
            return [merged, *msgs[1:]]
        # 兜底：inner 没给出前置 system 消息时，补一个
        return [Message(role=Role.SYSTEM, content=extra.lstrip()), *msgs]
