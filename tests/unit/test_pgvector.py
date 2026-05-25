"""Unit tests for PgVectorBackend (mock asyncpg — no real DB required)."""
from __future__ import annotations

import json
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Inject a fake asyncpg into sys.modules BEFORE the backend is imported so
# the module-level `import asyncpg` in pgvector.py succeeds without the package.
# ---------------------------------------------------------------------------
_fake_asyncpg = MagicMock()
_fake_asyncpg.Pool = MagicMock
_fake_asyncpg.Record = MagicMock
sys.modules.setdefault("asyncpg", _fake_asyncpg)

from agent_memory_manager.backends.pgvector import PgVectorBackend  # noqa: E402
from agent_memory_manager.models import MemoryRecord, MemoryType  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(
    content: str = "test content",
    session_id: str = "sess1",
    user_id: str | None = "user1",
    embedding: list[float] | None = None,
) -> MemoryRecord:
    r = MemoryRecord(
        session_id=session_id,
        user_id=user_id,
        content=content,
        memory_type=MemoryType.EPISODIC,
    )
    r.embedding = embedding or []
    return r


def _make_row(record: MemoryRecord) -> MagicMock:
    """Build a fake asyncpg.Record-like object from a MemoryRecord."""
    data = {
        "id": record.id,
        "session_id": record.session_id,
        "user_id": record.user_id,
        "memory_type": record.memory_type.value,
        "content": record.content,
        "source_ids": json.dumps(record.source_message_ids),
        "embedding": record.embedding or None,
        "importance": record.importance_score,
        "recency": record.recency_score,
        "keywords": json.dumps(record.keywords),
        "links": json.dumps(record.links),
        "created_at": record.created_at,
        "accessed_at": record.accessed_at,
        "metadata": json.dumps(record.metadata),
    }
    row = MagicMock()
    row.__getitem__ = lambda self, key: data[key]
    return row


def _make_pool_and_conn():
    """Return (pool, conn) where pool.acquire() is an async context manager."""
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="INSERT 0 1")
    conn.fetchrow = AsyncMock(return_value=None)
    conn.fetch = AsyncMock(return_value=[])
    conn.fetchval = AsyncMock(return_value=0)

    acquire_ctx = MagicMock()
    acquire_ctx.__aenter__ = AsyncMock(return_value=conn)
    acquire_ctx.__aexit__ = AsyncMock(return_value=False)

    pool = MagicMock()
    pool.acquire = MagicMock(return_value=acquire_ctx)
    pool.close = AsyncMock()

    return pool, conn


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def backend_with_pool():
    """Return a PgVectorBackend with its pool pre-injected (skips initialize)."""
    pool, conn = _make_pool_and_conn()
    backend = PgVectorBackend(dsn="postgresql://fake/db", vector_dim=3)
    backend._pool = pool
    return backend, pool, conn


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPgVectorBackend:
    def test_pool_conn_raises_before_initialize(self):
        backend = PgVectorBackend(dsn="postgresql://fake/db")
        with pytest.raises(RuntimeError, match="not initialized"):
            backend._pool_conn()

    @pytest.mark.asyncio
    async def test_save_inserts_record(self, backend_with_pool):
        backend, pool, conn = backend_with_pool
        record = _make_record(embedding=[0.1, 0.2, 0.3])
        returned_id = await backend.save(record)
        assert returned_id == record.id
        conn.execute.assert_called_once()
        sql = conn.execute.call_args[0][0]
        assert "INSERT INTO memories" in sql

    @pytest.mark.asyncio
    async def test_save_no_embedding_passes_none(self, backend_with_pool):
        backend, pool, conn = backend_with_pool
        record = _make_record(embedding=[])
        await backend.save(record)
        # embedding argument is the 7th positional arg (index 6 in call args[0][1:])
        args = conn.execute.call_args[0]
        # args[0] = sql, args[7] = emb (1-indexed: $7 maps to args[7])
        assert args[7] is None

    @pytest.mark.asyncio
    async def test_get_returns_none_when_not_found(self, backend_with_pool):
        backend, pool, conn = backend_with_pool
        conn.fetchrow = AsyncMock(return_value=None)
        result = await backend.get("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_returns_record_when_found(self, backend_with_pool):
        backend, pool, conn = backend_with_pool
        record = _make_record("hello world", embedding=[0.5, 0.5, 0.0])
        row = _make_row(record)
        conn.fetchrow = AsyncMock(return_value=row)

        result = await backend.get(record.id)
        assert result is not None
        assert result.id == record.id
        assert result.content == "hello world"

    @pytest.mark.asyncio
    async def test_list_by_session_returns_empty(self, backend_with_pool):
        backend, pool, conn = backend_with_pool
        conn.fetch = AsyncMock(return_value=[])
        results = await backend.list_by_session("no-session")
        assert results == []

    @pytest.mark.asyncio
    async def test_list_by_session_returns_records(self, backend_with_pool):
        backend, pool, conn = backend_with_pool
        records = [_make_record(f"msg {i}") for i in range(3)]
        conn.fetch = AsyncMock(return_value=[_make_row(r) for r in records])

        results = await backend.list_by_session("sess1")
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_list_by_user(self, backend_with_pool):
        backend, pool, conn = backend_with_pool
        records = [_make_record(f"u msg {i}", user_id="u42") for i in range(2)]
        conn.fetch = AsyncMock(return_value=[_make_row(r) for r in records])

        results = await backend.list_by_user("u42")
        assert len(results) == 2
        sql = conn.fetch.call_args[0][0]
        assert "user_id" in sql

    @pytest.mark.asyncio
    async def test_delete_returns_true_on_success(self, backend_with_pool):
        backend, pool, conn = backend_with_pool
        conn.execute = AsyncMock(return_value="DELETE 1")
        ok = await backend.delete("some-id")
        assert ok is True

    @pytest.mark.asyncio
    async def test_delete_returns_false_when_not_found(self, backend_with_pool):
        backend, pool, conn = backend_with_pool
        conn.execute = AsyncMock(return_value="DELETE 0")
        ok = await backend.delete("missing-id")
        assert ok is False

    @pytest.mark.asyncio
    async def test_delete_by_session_returns_count(self, backend_with_pool):
        backend, pool, conn = backend_with_pool
        conn.execute = AsyncMock(return_value="DELETE 5")
        count = await backend.delete_by_session("sess1")
        assert count == 5

    @pytest.mark.asyncio
    async def test_delete_by_user_returns_count(self, backend_with_pool):
        backend, pool, conn = backend_with_pool
        conn.execute = AsyncMock(return_value="DELETE 3")
        count = await backend.delete_by_user("user1")
        assert count == 3

    @pytest.mark.asyncio
    async def test_count_all(self, backend_with_pool):
        backend, pool, conn = backend_with_pool
        conn.fetchval = AsyncMock(return_value=42)
        total = await backend.count()
        assert total == 42

    @pytest.mark.asyncio
    async def test_count_by_session(self, backend_with_pool):
        backend, pool, conn = backend_with_pool
        conn.fetchval = AsyncMock(return_value=7)
        total = await backend.count("sess1")
        assert total == 7

    @pytest.mark.asyncio
    async def test_search_by_vector_returns_top_k(self, backend_with_pool):
        backend, pool, conn = backend_with_pool
        records = [_make_record(f"result {i}", embedding=[float(i), 0.0, 0.0]) for i in range(5)]
        conn.fetch = AsyncMock(return_value=[_make_row(r) for r in records])

        results = await backend.search_by_vector([1.0, 0.0, 0.0], top_k=3)
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_update_allowed_fields(self, backend_with_pool):
        backend, pool, conn = backend_with_pool
        conn.execute = AsyncMock(return_value="UPDATE 1")
        ok = await backend.update("some-id", {"content": "new content", "importance": 8.0})
        assert ok is True
        sql = conn.execute.call_args[0][0]
        assert "UPDATE memories SET" in sql

    @pytest.mark.asyncio
    async def test_update_disallowed_fields_returns_false(self, backend_with_pool):
        backend, pool, conn = backend_with_pool
        ok = await backend.update("some-id", {"session_id": "hack"})
        assert ok is False
        conn.execute.assert_not_called()

    def test_build_filters_empty(self):
        where, params = PgVectorBackend._build_filters(None)
        assert where == ""
        assert params == []

    def test_build_filters_session_and_user(self):
        where, params = PgVectorBackend._build_filters(
            {"session_id": "s1", "user_id": "u1"}, start_idx=2
        )
        assert "session_id" in where
        assert "user_id" in where
        assert "s1" in params
        assert "u1" in params

    @pytest.mark.asyncio
    async def test_close_calls_pool_close(self, backend_with_pool):
        backend, pool, conn = backend_with_pool
        pool.close = AsyncMock()
        await backend.close()
        pool.close.assert_called_once()
        assert backend._pool is None
