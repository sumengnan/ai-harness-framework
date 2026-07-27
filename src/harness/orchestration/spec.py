from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AgentSpec:
    name: str
    description: str          # 给主 agent 看的能力说明
    system_prompt: str        # 子 agent 的 system prompt
    tool_names: list[str]     # 该角色可用的工具名（从工具池选子集）


class AgentRoster:
    def __init__(self, specs: list[AgentSpec]) -> None:
        self._specs: dict[str, AgentSpec] = {s.name: s for s in specs}

    def get(self, name: str) -> AgentSpec | None:
        return self._specs.get(name)

    def names(self) -> list[str]:
        return list(self._specs.keys())

    def describe(self) -> str:
        lines = [f"- {s.name}：{s.description}" for s in self._specs.values()]
        return "可派发的子 agent 角色：\n" + "\n".join(lines)
