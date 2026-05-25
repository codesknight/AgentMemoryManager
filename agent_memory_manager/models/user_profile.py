"""UserProfile — aggregated cross-session memory for a single user."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class UserProfile:
    """Synthesized long-term profile for a user, aggregated across all sessions."""

    user_id: str
    facts: list[str] = field(default_factory=list)
    preferences: dict[str, str] = field(default_factory=dict)
    session_ids: list[str] = field(default_factory=list)
    total_memories: int = 0
    raw_summary: str = ""
    synthesized_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # ── serialisation ──────────────────────────────────────────────────────────

    def to_json(self) -> str:
        return json.dumps({
            "user_id": self.user_id,
            "facts": self.facts,
            "preferences": self.preferences,
            "session_ids": self.session_ids,
            "total_memories": self.total_memories,
            "raw_summary": self.raw_summary,
            "synthesized_at": self.synthesized_at.isoformat(),
        })

    @classmethod
    def from_json(cls, data: str) -> UserProfile:
        d = json.loads(data)
        return cls(
            user_id=d["user_id"],
            facts=d.get("facts", []),
            preferences=d.get("preferences", {}),
            session_ids=d.get("session_ids", []),
            total_memories=d.get("total_memories", 0),
            raw_summary=d.get("raw_summary", ""),
            synthesized_at=datetime.fromisoformat(d["synthesized_at"]),
        )
