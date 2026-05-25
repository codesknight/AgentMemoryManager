"""SQLite persistence for SemanticMemory knowledge graphs.

Stores each session's graph as a JSON blob in a lightweight SQLite table.
This avoids a Neo4j dependency while providing cross-process persistence.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

try:
    import aiosqlite
except ImportError as exc:
    raise ImportError(
        "Install 'aiosqlite' to use GraphStore: pip install aiosqlite"
    ) from exc

from agent_memory_manager.memory.semantic_memory import SemanticMemory

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS kg_graphs (
    session_id TEXT PRIMARY KEY,
    graph_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""


class GraphStore:
    """Async SQLite-backed persistence for SemanticMemory instances."""

    def __init__(self, db_path: str = "graph.db") -> None:
        self._db_path = str(Path(db_path))
        self._conn: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self._conn = await aiosqlite.connect(self._db_path)
        await self._conn.execute(_CREATE_TABLE)
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def save(self, graph: SemanticMemory) -> None:
        """Persist a SemanticMemory graph to SQLite."""
        from datetime import datetime, timezone
        assert self._conn, "GraphStore not initialized"
        data = json.dumps(graph.to_dict(), default=str)
        now = datetime.now(timezone.utc).isoformat()
        await self._conn.execute(
            """INSERT INTO kg_graphs (session_id, graph_json, updated_at)
               VALUES (?, ?, ?)
               ON CONFLICT(session_id) DO UPDATE SET graph_json=excluded.graph_json,
                                                     updated_at=excluded.updated_at""",
            (graph.session_id, data, now),
        )
        await self._conn.commit()

    async def load(self, session_id: str) -> SemanticMemory | None:
        """Load a SemanticMemory graph from SQLite. Returns None if not found."""
        assert self._conn, "GraphStore not initialized"
        async with self._conn.execute(
            "SELECT graph_json FROM kg_graphs WHERE session_id = ?", (session_id,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        try:
            return SemanticMemory.from_dict(json.loads(row[0]))
        except Exception as exc:
            logger.warning("Failed to load graph for session=%s: %s", session_id, exc)
            return None

    async def delete(self, session_id: str) -> bool:
        """Delete a session's graph. Returns True if a row was deleted."""
        assert self._conn, "GraphStore not initialized"
        cursor = await self._conn.execute(
            "DELETE FROM kg_graphs WHERE session_id = ?", (session_id,)
        )
        await self._conn.commit()
        return cursor.rowcount > 0

    async def list_sessions(self) -> list[str]:
        """Return all session IDs that have a persisted graph."""
        assert self._conn, "GraphStore not initialized"
        async with self._conn.execute(
            "SELECT session_id FROM kg_graphs ORDER BY updated_at DESC"
        ) as cursor:
            rows = await cursor.fetchall()
        return [r[0] for r in rows]
