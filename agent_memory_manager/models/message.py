from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    role: Role
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = Field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"role": self.role.value, "content": self.content}

    def token_estimate(self) -> int:
        """Rough token count: ~4 chars per token."""
        return len(self.content) // 4 + 1
