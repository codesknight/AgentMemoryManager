from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from agent_memory_manager.models import MemoryRecord, MemoryType
from .base import MemoryBackend

try:
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import (
        Distance,
        FieldCondition,
        Filter,
        MatchValue,
        PointStruct,
        UpdateStatus,
        VectorParams,
    )
except ImportError as exc:
    raise ImportError(
        "Install 'qdrant-client' to use QdrantBackend: pip install qdrant-client"
    ) from exc

_COLLECTION = "agent_memories"
_JSON_LIST_FIELDS = ("source_message_ids", "keywords", "links", "metadata")


def _to_payload(record: MemoryRecord) -> dict[str, Any]:
    return {
        "session_id": record.session_id,
        "user_id": record.user_id or "",
        "memory_type": record.memory_type.value,
        "content": record.content,
        "importance_score": record.importance_score,
        "recency_score": record.recency_score,
        "created_at": record.created_at.isoformat(),
        "accessed_at": record.accessed_at.isoformat(),
        "source_message_ids": json.dumps(record.source_message_ids),
        "keywords": json.dumps(record.keywords),
        "links": json.dumps(record.links),
        "metadata": json.dumps(record.metadata),
        "memory_id": record.id,  # stored separately since Qdrant uses UUID point IDs
    }


def _payload_to_record(
    point_id: str | int,
    payload: dict,
    vector: Optional[list[float]] = None,
) -> MemoryRecord:
    return MemoryRecord(
        id=payload.get("memory_id", str(point_id)),
        session_id=payload.get("session_id", ""),
        user_id=payload.get("user_id") or None,
        memory_type=MemoryType(payload.get("memory_type", "episodic")),
        content=payload.get("content", ""),
        embedding=vector,
        importance_score=float(payload.get("importance_score", 5.0)),
        recency_score=float(payload.get("recency_score", 1.0)),
        created_at=datetime.fromisoformat(payload["created_at"]),
        accessed_at=datetime.fromisoformat(payload["accessed_at"]),
        source_message_ids=json.loads(payload.get("source_message_ids", "[]")),
        keywords=json.loads(payload.get("keywords", "[]")),
        links=json.loads(payload.get("links", "[]")),
        metadata=json.loads(payload.get("metadata", "{}")),
    )


def _str_to_uuid(s: str) -> str:
    """Map an arbitrary string ID to a deterministic UUID for Qdrant."""
    try:
        uuid.UUID(s)
        return s
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, s))


class QdrantBackend(MemoryBackend):
    """Production-grade vector memory store backed by Qdrant.

    Supports three deployment modes:
    - **Local in-memory** (tests/dev): ``QdrantBackend(location=":memory:")``
    - **On-disk**:                    ``QdrantBackend(path="./qdrant_data")``
    - **Remote server**:              ``QdrantBackend(url="http://localhost:6333")``

    Qdrant's HNSW index delivers sub-millisecond approximate nearest-neighbour
    search at millions-of-vectors scale — suitable for large production deployments.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        path: Optional[str] = None,
        location: str = ":memory:",
        collection_name: str = _COLLECTION,
        vector_size: int = 1536,
        api_key: Optional[str] = None,
    ) -> None:
        self._url = url
        self._path = path
        self._location = location
        self._collection_name = collection_name
        self._vector_size = vector_size
        self._api_key = api_key
        self._client: Optional[AsyncQdrantClient] = None

    async def initialize(self) -> None:
        if self._url:
            self._client = AsyncQdrantClient(url=self._url, api_key=self._api_key)
        elif self._path:
            self._client = AsyncQdrantClient(path=self._path)
        else:
            self._client = AsyncQdrantClient(location=self._location)

        existing = await self._client.collection_exists(self._collection_name)
        if not existing:
            await self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=self._vector_size,
                    distance=Distance.COSINE,
                ),
            )

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None

    def _cli(self) -> AsyncQdrantClient:
        if self._client is None:
            raise RuntimeError(
                "QdrantBackend not initialized — call await backend.initialize()"
            )
        return self._client

    async def save(self, record: MemoryRecord) -> str:
        point_id = _str_to_uuid(record.id)
        payload = _to_payload(record)
        vector = record.embedding or [0.0] * self._vector_size

        await self._cli().upsert(
            collection_name=self._collection_name,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)],
        )
        return record.id

    async def get(self, memory_id: str) -> Optional[MemoryRecord]:
        point_id = _str_to_uuid(memory_id)
        results = await self._cli().retrieve(
            collection_name=self._collection_name,
            ids=[point_id],
            with_payload=True,
            with_vectors=True,
        )
        if not results:
            return None
        pt = results[0]
        return _payload_to_record(pt.id, pt.payload or {}, pt.vector)  # type: ignore[arg-type]

    async def search_by_vector(
        self,
        embedding: list[float],
        top_k: int = 10,
        filters: Optional[dict] = None,
    ) -> list[MemoryRecord]:
        qdrant_filter = self._build_filter(filters)
        results = await self._cli().search(
            collection_name=self._collection_name,
            query_vector=embedding,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
            with_vectors=True,
        )
        return [
            _payload_to_record(pt.id, pt.payload or {}, pt.vector)  # type: ignore[arg-type]
            for pt in results
        ]

    async def list_by_session(
        self,
        session_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryRecord]:
        results, _ = await self._cli().scroll(
            collection_name=self._collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
            ),
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        records = [
            _payload_to_record(pt.id, pt.payload or {}, pt.vector)  # type: ignore[arg-type]
            for pt in results
        ]
        records.sort(key=lambda r: r.created_at)
        return records

    async def update(self, memory_id: str, updates: dict) -> bool:
        existing = await self.get(memory_id)
        if not existing:
            return False
        for key, value in updates.items():
            if hasattr(existing, key):
                setattr(existing, key, value)
        await self.save(existing)
        return True

    async def delete(self, memory_id: str) -> bool:
        point_id = _str_to_uuid(memory_id)
        result = await self._cli().delete(
            collection_name=self._collection_name,
            points_selector=[point_id],
        )
        return result.status == UpdateStatus.COMPLETED

    async def delete_by_session(self, session_id: str) -> int:
        # Count first, then delete
        before = await self.count(session_id)
        await self._cli().delete(
            collection_name=self._collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
            ),
        )
        return before

    async def count(self, session_id: Optional[str] = None) -> int:
        if session_id is None:
            info = await self._cli().get_collection(self._collection_name)
            return info.points_count or 0
        result = await self._cli().count(
            collection_name=self._collection_name,
            count_filter=Filter(
                must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
            ),
        )
        return result.count

    @staticmethod
    def _build_filter(filters: Optional[dict]) -> Optional[Filter]:
        if not filters:
            return None
        conditions = []
        if sid := filters.get("session_id"):
            conditions.append(FieldCondition(key="session_id", match=MatchValue(value=sid)))
        if mtype := filters.get("memory_type"):
            v = mtype if isinstance(mtype, str) else mtype.value
            conditions.append(FieldCondition(key="memory_type", match=MatchValue(value=v)))
        return Filter(must=conditions) if conditions else None
