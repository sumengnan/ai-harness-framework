from __future__ import annotations

from dataclasses import dataclass, field

from .types import Message


@dataclass
class RunState:
    run_id: str
    messages: list[Message] = field(default_factory=list)
    step: int = 0

    def append(self, message: Message) -> None:
        self.messages.append(message)
