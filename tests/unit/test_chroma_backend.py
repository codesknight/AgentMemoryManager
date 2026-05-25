"""Unit tests for ChromaBackend — skipped if chromadb is not installed."""
import pytest

chromadb = pytest.importorskip("chromadb", reason="chromadb not installed")

from agent_memory_manager.backends.chroma import ChromaBackend
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
async def chroma():
    backend = ChromaBackend(ephemeral=True)
    await backend.initialize()
    yield backend
    await backend.close()


@pytest.mark.asyncio
async def test_save_and_get(chroma):
    record = _make_record()
    rid = await chroma.save(record)
    fetched = await chroma.get(rid)
    assert fetched is not None
    assert fetched.content == "test memory"
    assert fetched.session_id == "s1"


@pytest.mark.asyncio
async def test_save_idempotent(chroma):
    """Saving the same record twice should not raise."""
    record = _make_record()
    await chroma.save(record)
    record.content = "updated content"
    await chroma.save(record)
    fetched = await chroma.get(record.id)
    assert fetched is not None
    assert fetched.content == "updated content"


@pytest.mark.asyncio
async def test_get_nonexistent(chroma):
    result = await chroma.get("nonexistent-id")
    assert result is None


@pytest.mark.asyncio
async def test_search_by_vector(chroma):
    r1 = _make_record(content="Python programming")
    r1.embedding = [1.0, 0.0, 0.0, 0.0]
    await chroma.save(r1)

    r2 = _make_record(content="Cooking recipes")
    r2.embedding = [0.0, 1.0, 0.0, 0.0]
    await chroma.save(r2)

    results = await chroma.search_by_vector([1.0, 0.0, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0].content == "Python programming"


@pytest.mark.asyncio
async def test_search_with_session_filter(chroma):
    r1 = _make_record(session_id="session-A")
    r1.embedding = [1.0, 0.0, 0.0, 0.0]
    await chroma.save(r1)

    r2 = _make_record(session_id="session-B")
    r2.embedding = [1.0, 0.0, 0.0, 0.0]
    await chroma.save(r2)

    results = await chroma.search_by_vector(
        [1.0, 0.0, 0.0, 0.0], top_k=5, filters={"session_id": "session-A"}
    )
    assert all(r.session_id == "session-A" for r in results)


@pytest.mark.asyncio
async def test_list_by_session(chroma):
    for i in range(3):
        r = _make_record(session_id="sess-X", content=f"item {i}")
        r.embedding = [float(i) * 0.1, 0.0, 0.0, 0.0]
        await chroma.save(r)
    await chroma.save(_make_record(session_id="sess-Y"))

    results = await chroma.list_by_session("sess-X")
    assert len(results) == 3
    assert all(r.session_id == "sess-X" for r in results)


@pytest.mark.asyncio
async def test_update(chroma):
    record = _make_record()
    await chroma.save(record)

    ok = await chroma.update(record.id, {"content": "updated fact", "importance": 9.0})
    assert ok
    fetched = await chroma.get(record.id)
    assert fetched is not None
    assert fetched.content == "updated fact"


@pytest.mark.asyncio
async def test_delete(chroma):
    record = _make_record()
    await chroma.save(record)

    deleted = await chroma.delete(record.id)
    assert deleted
    assert await chroma.get(record.id) is None


@pytest.mark.asyncio
async def test_delete_nonexistent(chroma):
    result = await chroma.delete("no-such-id")
    assert not result


@pytest.mark.asyncio
async def test_delete_by_session(chroma):
    for _ in range(4):
        r = _make_record(session_id="del-session")
        r.embedding = [0.1, 0.2, 0.3, 0.4]
        await chroma.save(r)
    await chroma.save(_make_record(session_id="keep-session"))

    count = await chroma.delete_by_session("del-session")
    assert count == 4
    assert await chroma.count("del-session") == 0
    assert await chroma.count("keep-session") == 1


@pytest.mark.asyncio
async def test_count_total(chroma):
    for _ in range(3):
        r = _make_record()
        r.embedding = [0.1, 0.2, 0.3, 0.4]
        await chroma.save(r)
    assert await chroma.count() == 3
