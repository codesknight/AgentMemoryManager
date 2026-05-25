from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    EPISODIC = "episodic"      # Specific events / atomic facts
    SEMANTIC = "semantic"      # Entities and relations
    PROCEDURAL = "procedural"  # Task templates and tool patterns
    REFLECTION = "reflection"  # Higher-order insights synthesized from episodic memories


class MemoryRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    user_id: Optional[str] = None
    memory_type: MemoryType = MemoryType.EPISODIC
    content: str
    source_message_ids: list[str] = Field(default_factory=list)
    embedding: Optional[list[float]] = None
    importance_score: float = 5.0   # 1–10, scored by LLM
    recency_score: float = 1.0      # Computed via exponential decay at retrieval time
    keywords: list[str] = Field(default_factory=list)
    links: list[str] = Field(default_factory=list)  # Related memory IDs (Zettelkasten)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    accessed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = Field(default_factory=dict)

    def token_estimate(self) -> int:
        return len(self.content) // 4 + 1

    def touch(self) -> None:
        """Update accessed_at timestamp (affects recency scoring)."""
        self.accessed_at = datetime.now(timezone.utc)
