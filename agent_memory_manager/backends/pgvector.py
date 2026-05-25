"""PostgreSQL + pgvector backend for AgentMemoryManager (v2.0-C).

Install extras:
    pip install agent-memory-manager[pgvector]

Requires:
    PostgreSQL ≥ 14 with pgvector extension enabled:
        CREATE EXTENSION IF NOT EXISTS vector;

Connection string format:
    postgresql://user:password@host:5432/dbname
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from agent_memory_manager.models import MemoryRecord, MemoryType
from agent_memory_manager.utils.scoring import cosine_similarity

from .base import MemoryBackend

try:
    import asyncpg
except ImportError as exc:
    raise ImportError(
        "Install pgvector extras: pip install agent-memory-manager[pgvector]"
    ) from exc

_CREATE = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS memories (
    id           TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL,
    user_id      TEXT,
    memory_type  TEXT NOT NULL DEFAULT 'episodic',
    content      TEXT NOT NULL,
    source_ids   TEXT NOT NULL DEFAULT '[]',
    embedding    vector,
    importance   REAL NOT NULL DEFAULT 5.0,
    recency      REAL NOT NULL DEFAULT 1.0,
    keywords     TEXT NOT NULL DEFAULT '[]',
    links        TEXT NOT NULL DEFAULT '[]',
    created_at   TIMESTAMPTZ NOT NULL,
    accessed_at  TIMESTAMPTZ NOT NULL,
    metadata     TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_pg_session ON memories(session_id);
CREATE INDEX IF NOT EXISTS idx_pg_user    ON memories(user_id);
CREATE INDEX IF NOT EXISTS idx_pg_type    ON memories(memory_type);
"""


def _row_to_record(row: asyncpg.Record) -> MemoryRecord:
    emb = row["embedding"]
    if emb is not None and not isinstance(emb, list):
        emb = list(emb)
    return MemoryRecord(
        id=row["id"],
        session_id=row["session_id"],
        user_id=row["user_id"],
        memory_type=MemoryType(row["memory_type"]),
        content=row["content"],
        source_message_ids=json.loads(row["source_ids"]),
        embedding=emb,
        importance_score=row["importance"],
        recency_score=row["recency"],
        keywords=json.loads(row["keywords"]),
        links=json.loads(row["links"]),
        created_at=row["created_at"],
        accessed_at=row["accessed_at"],
        metadata=json.loads(row["metadata"]),
    )


class PgVectorBackend(MemoryBackend):
    """Production-grade backend backed by PostgreSQL + pgvector.

    Supports native ANN vector search via the pgvector ``<=>`` operator
    (cosine distance). Falls back to in-process cosine similarity when
    no embedding index is available.

    Args:
        dsn: PostgreSQL connection string.
        vector_dim: Embedding dimension (must match your embedder).
        pool_size: asyncpg connection pool size.
    """

    def __init__(
        self,
        dsn: str,
        vector_dim: int = 1536,
        pool_size: int = 5,
    ) -> None:
        self._dsn = dsn
        self._vector_dim = vector_dim
        self._pool_size = pool_size
        self._pool: Optional[asyncpg.Pool] = None

    async def initialize(self) -> None:
        self._pool = await asyncpg.create_pool(
            self._dsn, min_size=1, max_size=self._pool_size
        )
        async with self._pool.acquire() as conn:
            await conn.execute(_CREATE)
            # Try to create an IVFFlat index for ANN search
            try:
                await conn.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_pg_embedding
                    ON memories USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                    """
                )
            except Exception:
                pass  # pgvector < 0.5 or not enough data yet

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    def _pool_conn(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("PgVectorBackend not initialized — call await backend.initialize()")
        return self._pool

    async def save(self, record: MemoryRecord) -> str:
        emb = f"[{','.join(str(x) for x in record.embedding)}]" if record.embedding else None
        async with self._pool_conn().acquire() as conn:
            await conn.execute(
                """
                INSERT INTO memories
                    (id, session_id, user_id, memory_type, content, source_ids,
                     embedding, importance, recency, keywords, links,
                     created_at, accessed_at, metadata)
                VALUES ($1,$2,$3,$4,$5,$6,$7::vector,$8,$9,$10,$11,$12,$13,$14)
                ON CONFLICT (id) DO UPDATE SET
                    content=EXCLUDED.content,
                    embedding=EXCLUDED.embedding,
                    importance=EXCLUDED.importance,
                    recency=EXCLUDED.recency,
                    keywords=EXCLUDED.keywords,
                    links=EXCLUDED.links,
                    accessed_at=EXCLUDED.accessed_at,
                    metadata=EXCLUDED.metadata
                """,
                record.id, record.session_id, record.user_id,
                record.memory_type.value, record.content,
                json.dumps(record.source_message_ids),
                emb,
                record.importance_score, record.recency_score,
                json.dumps(record.keywords), json.dumps(record.links),
                record.created_at, record.accessed_at,
                json.dumps(record.metadata),
            )
        return record.id

    async def get(self, memory_id: str) -> Optional[MemoryRecord]:
        async with self._pool_conn().acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM memories WHERE id = $1", memory_id
            )
        return _row_to_record(row) if row else None

    async def search_by_vector(
        self,
        embedding: list[float],
        top_k: int = 10,
        filters: Optional[dict] = None,
    ) -> list[MemoryRecord]:
        where, params = self._build_filters(filters, start_idx=2)
        emb_str = f"[{','.join(str(x) for x in embedding)}]"
        # Use pgvector ANN operator for efficient similarity search
        query = f"""
            SELECT * FROM memories
            WHERE embedding IS NOT NULL {where}
            ORDER BY embedding <=> $1::vector
            LIMIT {top_k * 3}
        """
        async with self._pool_conn().acquire() as conn:
            rows = await conn.fetch(query, emb_str, *params)
        records = [_row_to_record(r) for r in rows]
        # Re-rank by full retrieval score in Python
        scored = [(r, cosine_similarity(r.embedding or [], embedding)) for r in records]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [r for r, _ in scored[:top_k]]

    async def list_by_session(
        self, session_id: str, limit: int = 100, offset: int = 0
    ) -> list[MemoryRecord]:
        async with self._pool_conn().acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM memories WHERE session_id=$1 ORDER BY created_at LIMIT $2 OFFSET $3",
                session_id, limit, offset,
            )
        return [_row_to_record(r) for r in rows]

    async def list_by_user(self, user_id: str, limit: int = 10_000) -> list[MemoryRecord]:
        async with self._pool_conn().acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM memories WHERE user_id=$1 ORDER BY created_at LIMIT $2",
                user_id, limit,
            )
        return [_row_to_record(r) for r in rows]

    async def update(self, memory_id: str, updates: dict) -> bool:
        allowed = {"content", "importance", "recency", "keywords",
                   "links", "accessed_at", "metadata", "embedding"}
        sets = {k: v for k, v in updates.items() if k in allowed}
        if not sets:
            return False
        for key in ("keywords", "links", "metadata"):
            if key in sets and not isinstance(sets[key], str):
                sets[key] = json.dumps(sets[key])
        if "embedding" in sets and sets["embedding"] is not None:
            sets["embedding"] = f"[{','.join(str(x) for x in sets['embedding'])}]::vector"

        cols = list(sets.keys())
        vals = list(sets.values())
        clause = ", ".join(f"{c}=${i+1}" for i, c in enumerate(cols))
        async with self._pool_conn().acquire() as conn:
            result = await conn.execute(
                f"UPDATE memories SET {clause} WHERE id=${len(cols)+1}",
                *vals, memory_id,
            )
        return result != "UPDATE 0"

    async def delete(self, memory_id: str) -> bool:
        async with self._pool_conn().acquire() as conn:
            result = await conn.execute(
                "DELETE FROM memories WHERE id=$1", memory_id
            )
        return result != "DELETE 0"

    async def delete_by_session(self, session_id: str) -> int:
        async with self._pool_conn().acquire() as conn:
            result = await conn.execute(
                "DELETE FROM memories WHERE session_id=$1", session_id
            )
        return int(result.split()[-1])

    async def delete_by_user(self, user_id: str) -> int:
        async with self._pool_conn().acquire() as conn:
            result = await conn.execute(
                "DELETE FROM memories WHERE user_id=$1", user_id
            )
        return int(result.split()[-1])

    async def count(self, session_id: Optional[str] = None) -> int:
        if session_id:
            async with self._pool_conn().acquire() as conn:
                return await conn.fetchval(
                    "SELECT COUNT(*) FROM memories WHERE session_id=$1", session_id
                )
        async with self._pool_conn().acquire() as conn:
            return await conn.fetchval("SELECT COUNT(*) FROM memories")

    @staticmethod
    def _build_filters(filters: Optional[dict], start_idx: int = 1) -> tuple[str, list]:
        if not filters:
            return "", []
        clauses, params = [], []
        idx = start_idx
        if sid := filters.get("session_id"):
            clauses.append(f"AND session_id = ${idx}")
            params.append(sid)
            idx += 1
        if uid := filters.get("user_id"):
            clauses.append(f"AND user_id = ${idx}")
            params.append(uid)
            idx += 1
        if mtype := filters.get("memory_type"):
            clauses.append(f"AND memory_type = ${idx}")
            params.append(mtype if isinstance(mtype, str) else mtype.value)
        return " ".join(clauses), params
