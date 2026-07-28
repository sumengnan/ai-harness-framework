from __future__ import annotations

from dataclasses import dataclass, field

from .types import Message


@dataclass
class RunState:
    run_id: str
    messages: list[Message] = field(default_factory=list)
    step: int = 0
    # 已消耗预算。跟着快照走，resume 时装回 BudgetTracker，
    # 否则续跑等于把预算重新给满一份。
    tokens_used: int = 0
    wall_seconds_used: float = 0.0

    def append(self, message: Message) -> None:
        self.messages.append(message)
