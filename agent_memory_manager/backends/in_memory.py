from __future__ import annotations

from typing import Optional

from agent_memory_manager.models import MemoryRecord
from agent_memory_manager.utils.scoring import cosine_similarity

from .base import MemoryBackend


class InMemoryBackend(MemoryBackend):
    """In-process memory store backed by a plain dict.

    Zero dependencies, instant start — ideal for unit tests and quick demos.
    Not persistent: all data is lost when the process exits.
    """

    def __init__(self) -> None:
        self._store: dict[str, MemoryRecord] = {}

    async def save(self, record: MemoryRecord) -> str:
        self._store[record.id] = record
        return record.id

    async def get(self, memory_id: str) -> Optional[MemoryRecord]:
        return self._store.get(memory_id)

    async def search_by_vector(
        self,
        embedding: list[float],
        top_k: int = 10,
        filters: Optional[dict] = None,
    ) -> list[MemoryRecord]:
        candidates = list(self._store.values())

        if filters:
            if sid := filters.get("session_id"):
                candidates = [r for r in candidates if r.session_id == sid]
            if uid := filters.get("user_id"):
                candidates = [r for r in candidates if r.user_id == uid]
            if mtype := filters.get("memory_type"):
                candidates = [r for r in candidates if r.memory_type == mtype]

        scored = [
            (r, cosine_similarity(r.embedding or [], embedding))
            for r in candidates
            if r.embedding
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [r for r, _ in scored[:top_k]]

    async def list_by_session(
        self,
        session_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryRecord]:
        results = [r for r in self._store.values() if r.session_id == session_id]
        results.sort(key=lambda r: r.created_at)
        return results[offset : offset + limit]

    async def update(self, memory_id: str, updates: dict) -> bool:
        record = self._store.get(memory_id)
        if not record:
            return False
        for key, value in updates.items():
            if hasattr(record, key):
                setattr(record, key, value)
        return True

    async def delete(self, memory_id: str) -> bool:
        return self._store.pop(memory_id, None) is not None

    async def delete_by_session(self, session_id: str) -> int:
        to_delete = [k for k, v in self._store.items() if v.session_id == session_id]
        for key in to_delete:
            del self._store[key]
        return len(to_delete)

    async def count(self, session_id: Optional[str] = None) -> int:
        if session_id is None:
            return len(self._store)
        return sum(1 for r in self._store.values() if r.session_id == session_id)

    async def list_by_user(self, user_id: str, limit: int = 10_000) -> list[MemoryRecord]:
        results = [r for r in self._store.values() if r.user_id == user_id]
        results.sort(key=lambda r: r.created_at)
        return results[:limit]

    async def delete_by_user(self, user_id: str) -> int:
        to_delete = [k for k, v in self._store.items() if v.user_id == user_id]
        for key in to_delete:
            del self._store[key]
        return len(to_delete)
