"""Unit tests for storage backends (no LLM calls needed)."""
import pytest

from agent_memory_manager.backends.in_memory import InMemoryBackend
from agent_memory_manager.backends.sqlite import SQLiteBackend
from agent_memory_manager.models import MemoryRecord


@pytest.fixture
def in_memory_backend():
    return InMemoryBackend()


@pytest.fixture
async def sqlite_backend():
    backend = SQLiteBackend(":memory:")
    await backend.initialize()
    yield backend
    await backend.close()


def _make_record(session_id: str = "s1", content: str = "test fact") -> MemoryRecord:
    r = MemoryRecord(session_id=session_id, content=content)
    r.embedding = [1.0, 0.0, 0.0]
    return r


# ────── InMemoryBackend tests ──────

@pytest.mark.asyncio
async def test_in_memory_save_get(in_memory_backend):
    record = _make_record()
    rid = await in_memory_backend.save(record)
    fetched = await in_memory_backend.get(rid)
    assert fetched is not None
    assert fetched.content == "test fact"


@pytest.mark.asyncio
async def test_in_memory_delete(in_memory_backend):
    record = _make_record()
    await in_memory_backend.save(record)
    deleted = await in_memory_backend.delete(record.id)
    assert deleted
    assert await in_memory_backend.get(record.id) is None


@pytest.mark.asyncio
async def test_in_memory_search_by_vector(in_memory_backend):
    r1 = _make_record(content="Python facts")
    r1.embedding = [1.0, 0.0]
    await in_memory_backend.save(r1)

    r2 = _make_record(content="Unrelated")
    r2.embedding = [0.0, 1.0]
    await in_memory_backend.save(r2)

    results = await in_memory_backend.search_by_vector([1.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0].content == "Python facts"


@pytest.mark.asyncio
async def test_in_memory_delete_by_session(in_memory_backend):
    for i in range(3):
        await in_memory_backend.save(_make_record(session_id="session-A"))
    await in_memory_backend.save(_make_record(session_id="session-B"))

    deleted = await in_memory_backend.delete_by_session("session-A")
    assert deleted == 3
    assert await in_memory_backend.count("session-A") == 0
    assert await in_memory_backend.count("session-B") == 1


# ────── SQLiteBackend tests ──────

@pytest.mark.asyncio
async def test_sqlite_save_get(sqlite_backend):
    record = _make_record()
    rid = await sqlite_backend.save(record)
    fetched = await sqlite_backend.get(rid)
    assert fetched is not None
    assert fetched.content == "test fact"


@pytest.mark.asyncio
async def test_sqlite_update(sqlite_backend):
    record = _make_record()
    await sqlite_backend.save(record)
    ok = await sqlite_backend.update(record.id, {"content": "updated"})
    assert ok
    fetched = await sqlite_backend.get(record.id)
    assert fetched is not None
    assert fetched.content == "updated"


@pytest.mark.asyncio
async def test_sqlite_delete_by_session(sqlite_backend):
    for _ in range(5):
        await sqlite_backend.save(_make_record(session_id="sess-X"))
    count = await sqlite_backend.delete_by_session("sess-X")
    assert count == 5
    assert await sqlite_backend.count("sess-X") == 0
