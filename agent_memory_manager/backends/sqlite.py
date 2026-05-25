from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from agent_memory_manager.models import MemoryRecord, MemoryType
from agent_memory_manager.utils.scoring import cosine_similarity

from .base import MemoryBackend

try:
    import aiosqlite
except ImportError as exc:
    raise ImportError("Install 'aiosqlite' to use SQLiteBackend: pip install aiosqlite") from exc

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS memories (
    id           TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL,
    user_id      TEXT,
    memory_type  TEXT NOT NULL DEFAULT 'episodic',
    content      TEXT NOT NULL,
    source_ids   TEXT NOT NULL DEFAULT '[]',
    embedding    TEXT,
    importance   REAL NOT NULL DEFAULT 5.0,
    recency      REAL NOT NULL DEFAULT 1.0,
    keywords     TEXT NOT NULL DEFAULT '[]',
    links        TEXT NOT NULL DEFAULT '[]',
    created_at   TEXT NOT NULL,
    accessed_at  TEXT NOT NULL,
    metadata     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_session ON memories(session_id);
CREATE INDEX IF NOT EXISTS idx_type    ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_user    ON memories(user_id);
"""


def _row_to_record(row: aiosqlite.Row) -> MemoryRecord:
    return MemoryRecord(
        id=row["id"],
        session_id=row["session_id"],
        user_id=row["user_id"],
        memory_type=MemoryType(row["memory_type"]),
        content=row["content"],
        source_message_ids=json.loads(row["source_ids"]),
        embedding=json.loads(row["embedding"]) if row["embedding"] else None,
        importance_score=row["importance"],
        recency_score=row["recency"],
        keywords=json.loads(row["keywords"]),
        links=json.loads(row["links"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        accessed_at=datetime.fromisoformat(row["accessed_at"]),
        metadata=json.loads(row["metadata"]),
    )


class SQLiteBackend(MemoryBackend):
    """Persistent memory store backed by SQLite + in-process vector search.

    Suitable for single-process deployments and local development.
    Vector search is done in Python (exact cosine similarity),
    which is acceptable for sessions with < 50,000 memories.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def initialize(self) -> None:
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_CREATE_TABLE)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    def _conn(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("SQLiteBackend not initialized — call await backend.initialize()")
        return self._db

    async def save(self, record: MemoryRecord) -> str:
        await self._conn().execute(
            """
            INSERT OR REPLACE INTO memories
            (id, session_id, user_id, memory_type, content, source_ids,
             embedding, importance, recency, keywords, links,
             created_at, accessed_at, metadata)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                record.id,
                record.session_id,
                record.user_id,
                record.memory_type.value,
                record.content,
                json.dumps(record.source_message_ids),
                json.dumps(record.embedding) if record.embedding else None,
                record.importance_score,
                record.recency_score,
                json.dumps(record.keywords),
                json.dumps(record.links),
                record.created_at.isoformat(),
                record.accessed_at.isoformat(),
                json.dumps(record.metadata),
            ),
        )
        await self._conn().commit()
        return record.id

    async def get(self, memory_id: str) -> Optional[MemoryRecord]:
        async with self._conn().execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return _row_to_record(row) if row else None

    async def search_by_vector(
        self,
        embedding: list[float],
        top_k: int = 10,
        filters: Optional[dict] = None,
    ) -> list[MemoryRecord]:
        where, params = self._build_filters(filters)
        async with self._conn().execute(
            f"SELECT * FROM memories{where}", params
        ) as cursor:
            rows = await cursor.fetchall()

        records = [_row_to_record(r) for r in rows]
        scored = [
            (r, cosine_similarity(r.embedding or [], embedding))
            for r in records
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
        async with self._conn().execute(
            "SELECT * FROM memories WHERE session_id = ? ORDER BY created_at LIMIT ? OFFSET ?",
            (session_id, limit, offset),
        ) as cursor:
            return [_row_to_record(r) for r in await cursor.fetchall()]

    async def update(self, memory_id: str, updates: dict) -> bool:
        allowed = {
            "content", "importance", "recency", "keywords",
            "links", "accessed_at", "metadata", "embedding",
        }
        sets = {k: v for k, v in updates.items() if k in allowed}
        if not sets:
            return False

        # Serialize complex types
        for key in ("keywords", "links", "metadata", "embedding"):
            if key in sets and sets[key] is not None:
                sets[key] = json.dumps(sets[key])
        if "accessed_at" in sets and isinstance(sets["accessed_at"], datetime):
            sets["accessed_at"] = sets["accessed_at"].isoformat()

        clause = ", ".join(f"{k} = ?" for k in sets)
        await self._conn().execute(
            f"UPDATE memories SET {clause} WHERE id = ?",
            (*sets.values(), memory_id),
        )
        await self._conn().commit()
        return True

    async def delete(self, memory_id: str) -> bool:
        cursor = await self._conn().execute(
            "DELETE FROM memories WHERE id = ?", (memory_id,)
        )
        await self._conn().commit()
        return cursor.rowcount > 0

    async def delete_by_session(self, session_id: str) -> int:
        cursor = await self._conn().execute(
            "DELETE FROM memories WHERE session_id = ?", (session_id,)
        )
        await self._conn().commit()
        return cursor.rowcount

    async def list_by_user(self, user_id: str, limit: int = 10_000) -> list[MemoryRecord]:
        async with self._conn().execute(
            "SELECT * FROM memories WHERE user_id = ? ORDER BY created_at LIMIT ?",
            (user_id, limit),
        ) as cursor:
            return [_row_to_record(r) for r in await cursor.fetchall()]

    async def delete_by_user(self, user_id: str) -> int:
        cursor = await self._conn().execute(
            "DELETE FROM memories WHERE user_id = ?", (user_id,)
        )
        await self._conn().commit()
        return cursor.rowcount

    async def count(self, session_id: Optional[str] = None) -> int:
        if session_id:
            async with self._conn().execute(
                "SELECT COUNT(*) FROM memories WHERE session_id = ?", (session_id,)
            ) as cur:
                row = await cur.fetchone()
        else:
            async with self._conn().execute("SELECT COUNT(*) FROM memories") as cur:
                row = await cur.fetchone()
        return row[0] if row else 0

    @staticmethod
    def _build_filters(filters: Optional[dict]) -> tuple[str, list]:
        if not filters:
            return "", []
        clauses, params = [], []
        if sid := filters.get("session_id"):
            clauses.append("session_id = ?")
            params.append(sid)
        if mtype := filters.get("memory_type"):
            clauses.append("memory_type = ?")
            params.append(mtype if isinstance(mtype, str) else mtype.value)
        if uid := filters.get("user_id"):
            clauses.append("user_id = ?")
            params.append(uid)
        if not clauses:
            return "", []
        return " WHERE " + " AND ".join(clauses), params
