from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from agent_memory_manager.models import MemoryRecord


class MemoryBackend(ABC):
    """Unified storage abstraction for all backend implementations."""

    @abstractmethod
    async def save(self, record: MemoryRecord) -> str:
        """Persist a memory record; return its ID."""
        ...

    @abstractmethod
    async def get(self, memory_id: str) -> Optional[MemoryRecord]:
        ...

    @abstractmethod
    async def search_by_vector(
        self,
        embedding: list[float],
        top_k: int = 10,
        filters: Optional[dict] = None,
    ) -> list[MemoryRecord]:
        """Return top_k records ordered by cosine similarity (descending)."""
        ...

    @abstractmethod
    async def list_by_session(
        self,
        session_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryRecord]:
        ...

    @abstractmethod
    async def update(self, memory_id: str, updates: dict) -> bool:
        ...

    @abstractmethod
    async def delete(self, memory_id: str) -> bool:
        ...

    @abstractmethod
    async def delete_by_session(self, session_id: str) -> int:
        """Delete all memories for a session. Returns count deleted."""
        ...

    @abstractmethod
    async def count(self, session_id: Optional[str] = None) -> int:
        ...

    # Optional lifecycle hooks — backends may override
    async def initialize(self) -> None:
        """Called once before first use (e.g. create tables)."""

    async def close(self) -> None:
        """Release resources (e.g. close DB connections)."""
