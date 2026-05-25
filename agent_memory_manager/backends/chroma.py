from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from agent_memory_manager.models import MemoryRecord, MemoryType
from agent_memory_manager.utils.scoring import cosine_similarity

from .base import MemoryBackend

try:
    import chromadb
    from chromadb.config import Settings
except ImportError as exc:
    raise ImportError(
        "Install 'chromadb' to use ChromaBackend: pip install chromadb"
    ) from exc

_COLLECTION = "agent_memories"

# Chroma metadata only supports str/int/float/bool — complex types are JSON-encoded
_JSON_FIELDS = ("source_message_ids", "keywords", "links", "metadata")


def _serialize_meta(record: MemoryRecord) -> dict[str, Any]:
    """Flatten MemoryRecord fields into a Chroma-compatible metadata dict."""
    return {
        "session_id": record.session_id,
        "user_id": record.user_id or "",
        "memory_type": record.memory_type.value,
        "importance_score": record.importance_score,
        "recency_score": record.recency_score,
        "created_at": record.created_at.isoformat(),
        "accessed_at": record.accessed_at.isoformat(),
        "source_message_ids": json.dumps(record.source_message_ids),
        "keywords": json.dumps(record.keywords),
        "links": json.dumps(record.links),
        "metadata": json.dumps(record.metadata),
    }


def _meta_to_record(
    id_: str,
    document: str,
    meta: dict,
    embedding: Optional[list[float]] = None,
) -> MemoryRecord:
    return MemoryRecord(
        id=id_,
        session_id=meta.get("session_id", ""),
        user_id=meta.get("user_id") or None,
        memory_type=MemoryType(meta.get("memory_type", "episodic")),
        content=document,
        embedding=embedding,
        importance_score=float(meta.get("importance_score", 5.0)),
        recency_score=float(meta.get("recency_score", 1.0)),
        created_at=datetime.fromisoformat(meta["created_at"]),
        accessed_at=datetime.fromisoformat(meta["accessed_at"]),
        source_message_ids=json.loads(meta.get("source_message_ids", "[]")),
        keywords=json.loads(meta.get("keywords", "[]")),
        links=json.loads(meta.get("links", "[]")),
        metadata=json.loads(meta.get("metadata", "{}")),
    )


class ChromaBackend(MemoryBackend):
    """Vector memory store backed by ChromaDB.

    Supports two modes:
    - **Persistent** (default): ``ChromaBackend(path="./chroma_db")``
    - **Ephemeral** (tests):   ``ChromaBackend(ephemeral=True)``
    - **Remote**:              ``ChromaBackend(host="localhost", port=8000)``

    Chroma handles vector indexing (HNSW) internally, so search is fast
    even at large scales without a dedicated vector DB server.
    """

    def __init__(
        self,
        path: str = "./chroma_db",
        ephemeral: bool = False,
        host: Optional[str] = None,
        port: int = 8000,
        collection_name: str = _COLLECTION,
    ) -> None:
        self._path = path
        self._ephemeral = ephemeral
        self._host = host
        self._port = port
        self._collection_name = collection_name
        self._client: Any = None
        self._collection: Any = None

    async def initialize(self) -> None:
        if self._host:
            self._client = chromadb.HttpClient(host=self._host, port=self._port)
        elif self._ephemeral:
            self._client = chromadb.EphemeralClient()
        else:
            self._client = chromadb.PersistentClient(path=self._path)

        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    async def close(self) -> None:
        self._client = None
        self._collection = None

    def _col(self):
        if self._collection is None:
            raise RuntimeError(
                "ChromaBackend not initialized — call await backend.initialize()"
            )
        return self._collection

    async def save(self, record: MemoryRecord) -> str:
        meta = _serialize_meta(record)
        existing_ids = self._col().get(ids=[record.id])["ids"]
        if existing_ids:
            # Update in place
            kwargs: dict = dict(
                ids=[record.id],
                documents=[record.content],
                metadatas=[meta],
            )
            if record.embedding:
                kwargs["embeddings"] = [record.embedding]
            self._col().update(**kwargs)
        else:
            kwargs = dict(
                ids=[record.id],
                documents=[record.content],
                metadatas=[meta],
            )
            if record.embedding:
                kwargs["embeddings"] = [record.embedding]
            self._col().add(**kwargs)
        return record.id

    async def get(self, memory_id: str) -> Optional[MemoryRecord]:
        result = self._col().get(
            ids=[memory_id],
            include=["documents", "metadatas", "embeddings"],
        )
        if not result["ids"]:
            return None
        return _meta_to_record(
            id_=result["ids"][0],
            document=result["documents"][0],
            meta=result["metadatas"][0],
            embedding=result["embeddings"][0] if result.get("embeddings") else None,
        )

    async def search_by_vector(
        self,
        embedding: list[float],
        top_k: int = 10,
        filters: Optional[dict] = None,
    ) -> list[MemoryRecord]:
        where: Optional[dict] = None
        if filters:
            conditions = []
            if sid := filters.get("session_id"):
                conditions.append({"session_id": {"$eq": sid}})
            if mtype := filters.get("memory_type"):
                v = mtype if isinstance(mtype, str) else mtype.value
                conditions.append({"memory_type": {"$eq": v}})
            if len(conditions) == 1:
                where = conditions[0]
            elif len(conditions) > 1:
                where = {"$and": conditions}

        query_kwargs: dict = dict(
            query_embeddings=[embedding],
            n_results=min(top_k, max(1, self._col().count())),
            include=["documents", "metadatas", "embeddings", "distances"],
        )
        if where:
            query_kwargs["where"] = where

        result = self._col().query(**query_kwargs)
        records = []
        for i, rid in enumerate(result["ids"][0]):
            records.append(
                _meta_to_record(
                    id_=rid,
                    document=result["documents"][0][i],
                    meta=result["metadatas"][0][i],
                    embedding=(
                        result["embeddings"][0][i]
                        if result.get("embeddings")
                        else None
                    ),
                )
            )
        return records

    async def list_by_session(
        self,
        session_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryRecord]:
        result = self._col().get(
            where={"session_id": {"$eq": session_id}},
            include=["documents", "metadatas", "embeddings"],
            limit=limit,
            offset=offset,
        )
        records = [
            _meta_to_record(
                id_=rid,
                document=result["documents"][i],
                meta=result["metadatas"][i],
                embedding=(
                    result["embeddings"][i]
                    if result.get("embeddings")
                    else None
                ),
            )
            for i, rid in enumerate(result["ids"])
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
        meta = _serialize_meta(existing)
        kwargs: dict = dict(
            ids=[memory_id],
            documents=[existing.content],
            metadatas=[meta],
        )
        if existing.embedding:
            kwargs["embeddings"] = [existing.embedding]
        self._col().update(**kwargs)
        return True

    async def delete(self, memory_id: str) -> bool:
        existing = self._col().get(ids=[memory_id])["ids"]
        if not existing:
            return False
        self._col().delete(ids=[memory_id])
        return True

    async def delete_by_session(self, session_id: str) -> int:
        result = self._col().get(
            where={"session_id": {"$eq": session_id}},
            include=[],
        )
        ids = result["ids"]
        if ids:
            self._col().delete(ids=ids)
        return len(ids)

    async def count(self, session_id: Optional[str] = None) -> int:
        if session_id is None:
            return self._col().count()
        result = self._col().get(
            where={"session_id": {"$eq": session_id}},
            include=[],
        )
        return len(result["ids"])
