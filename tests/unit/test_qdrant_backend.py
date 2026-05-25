"""Unit tests for QdrantBackend — skipped if qdrant-client is not installed."""
import pytest

qdrant_client = pytest.importorskip("qdrant_client", reason="qdrant-client not installed")

from agent_memory_manager.backends.qdrant import QdrantBackend
from agent_memory_manager.models import MemoryRecord, MemoryType


def _make_record(
    session_id: str = "s1",
    content: str = "test memory",
    importance: float = 7.0,
) -> MemoryRecord:
    r = MemoryRecord(session_id=session_id, content=content, importance_score=importance)
    r.embedding = [0.1, 0.2, 0.3, 0.4]
    return r


@pytest.fixture
async def qdrant():
    backend = QdrantBackend(location=":memory:", vector_size=4)
    await backend.initialize()
    yield backend
    await backend.close()


@pytest.mark.asyncio
async def test_save_and_get(qdrant):
    record = _make_record()
    rid = await qdrant.save(record)
    fetched = await qdrant.get(rid)
    assert fetched is not None
    assert fetched.content == "test memory"
    assert fetched.session_id == "s1"


@pytest.mark.asyncio
async def test_get_nonexistent(qdrant):
    result = await qdrant.get("00000000-0000-0000-0000-000000000000")
    assert result is None


@pytest.mark.asyncio
async def test_search_by_vector(qdrant):
    r1 = _make_record(content="Python programming")
    r1.embedding = [1.0, 0.0, 0.0, 0.0]
    await qdrant.save(r1)

    r2 = _make_record(content="Cooking recipes")
    r2.embedding = [0.0, 1.0, 0.0, 0.0]
    await qdrant.save(r2)

    results = await qdrant.search_by_vector([1.0, 0.0, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0].content == "Python programming"


@pytest.mark.asyncio
async def test_session_filter(qdrant):
    r1 = _make_record(session_id="A")
    r1.embedding = [1.0, 0.0, 0.0, 0.0]
    await qdrant.save(r1)

    r2 = _make_record(session_id="B")
    r2.embedding = [1.0, 0.0, 0.0, 0.0]
    await qdrant.save(r2)

    results = await qdrant.search_by_vector(
        [1.0, 0.0, 0.0, 0.0], top_k=5, filters={"session_id": "A"}
    )
    assert all(r.session_id == "A" for r in results)


@pytest.mark.asyncio
async def test_list_by_session(qdrant):
    for i in range(3):
        r = _make_record(session_id="sess-X", content=f"item {i}")
        r.embedding = [float(i) * 0.25, 0.0, 0.0, 0.0]
        await qdrant.save(r)

    results = await qdrant.list_by_session("sess-X")
    assert len(results) == 3
    assert all(r.session_id == "sess-X" for r in results)


@pytest.mark.asyncio
async def test_update(qdrant):
    record = _make_record()
    await qdrant.save(record)
    ok = await qdrant.update(record.id, {"content": "updated"})
    assert ok
    fetched = await qdrant.get(record.id)
    assert fetched is not None
    assert fetched.content == "updated"


@pytest.mark.asyncio
async def test_delete(qdrant):
    record = _make_record()
    await qdrant.save(record)
    deleted = await qdrant.delete(record.id)
    assert deleted
    assert await qdrant.get(record.id) is None


@pytest.mark.asyncio
async def test_delete_by_session(qdrant):
    for _ in range(3):
        r = _make_record(session_id="del-me")
        r.embedding = [0.1, 0.2, 0.3, 0.4]
        await qdrant.save(r)

    count = await qdrant.delete_by_session("del-me")
    assert count == 3
    assert await qdrant.count("del-me") == 0


@pytest.mark.asyncio
async def test_count(qdrant):
    for _ in range(4):
        r = _make_record()
        r.embedding = [0.1, 0.2, 0.3, 0.4]
        await qdrant.save(r)
    assert await qdrant.count() == 4
    assert await qdrant.count("s1") == 4
