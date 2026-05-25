from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class Entity(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    name: str
    entity_type: str  # person / place / concept / organization / etc.
    attributes: dict = Field(default_factory=dict)
    embedding: Optional[list[float]] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Relation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    subject_id: str
    predicate: str
    object_id: str
    confidence: float = 1.0
    # Temporal awareness: None means the relation is still valid
    valid_from: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    valid_to: Optional[datetime] = None

    @property
    def is_current(self) -> bool:
        return self.valid_to is None
