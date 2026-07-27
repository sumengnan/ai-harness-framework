# src/harness/memory/record.py
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class MemType(str, Enum):
    EPISODIC = "episodic"      # 发生过什么（对话、事件）
    SEMANTIC = "semantic"      # 提炼的事实/偏好
    PROCEDURAL = "procedural"  # 学到的做事方法


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MemoryRecord:
    # 身份/隔离
    owner_id: str
    kind: str
    # 分型
    mem_type: MemType
    # 内容
    text: str
    embedding: list[float] = field(default_factory=list)  # 写入必填；检索回读为空
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    # 更新/矛盾（SP3 填；SP1 仅建列）
    entity_key: str = ""       # 空=哨兵
    version: int = 1
    superseded: int = 0        # 0/1
    # 检索打分（SP2 用；SP1 给默认）
    importance: float = 0.5
    created_at: str = field(default_factory=_now_iso)
    last_accessed_at: str = field(default_factory=_now_iso)
    access_count: int = 0
    # 生命周期（SP4 用；SP1 恒 0）
    expires_at: int = 0        # unix 秒；0=永不
    # 溯源
    source: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class MemoryFilter:
    owner_id: str
    kind: str | None = None
    mem_type: str | None = None
    include_superseded: bool = False
    include_expired: bool = False


@dataclass
class MemoryHit:
    record: MemoryRecord
    distance: float
