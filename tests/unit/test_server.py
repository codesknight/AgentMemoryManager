"""Unit tests for the REST API server (v2.0-B)."""
import json
import pytest
from unittest.mock import AsyncMock
from httpx import AsyncClient, ASGITransport

from agent_memory_manager.backends.in_memory import InMemoryBackend
from agent_memory_manager.manager import MemoryManager
from agent_memory_manager.models import MemoryRecord, MemoryType
from agent_memory_manager.server import create_app
from agent_memory_manager.strategies.sliding_window import SlidingWindowStrategy


def _mock_embedder():
    e = AsyncMock()
    e.embed = AsyncMock(return_value=[0.5, 0.5, 0.0, 0.0])
    e.dimensions = 4
    return e


def _mock_llm():
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=json.dumps({
        "facts": ["User is a backend engineer."],
        "preferences": {"language": "Python"},
        "raw_summary": "Backend engineer.",
    }))
    return llm


@pytest.fixture
async def client():
    manager = MemoryManager(
        backend=InMemoryBackend(),
        strategy=SlidingWindowStrategy(window_size=10),
        llm=_mock_llm(),
        embedder=_mock_embedder(),
        enable_graph=False,
    )
    app = create_app(manager)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac, manager


# ── Session endpoints ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_add_messages(client):
    ac, _ = client
    resp = await ac.post("/sessions/s1/add", json={
        "messages": [{"role": "user", "content": "Hello, I am Sam."}],
        "user_id": "u-1",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "added" in data
    assert isinstance(data["added"], list)


@pytest.mark.asyncio
async def test_search_memories(client):
    ac, manager = client
    r = MemoryRecord(session_id="s1", content="Sam is an engineer")
    r.embedding = [0.5, 0.5, 0.0, 0.0]
    await manager._backend.save(r)

    resp = await ac.post("/sessions/s1/search", json={"query": "engineer", "top_k": 5})
    assert resp.status_code == 200
    data = resp.json()
    assert "records" in data
    assert "scores" in data


@pytest.mark.asyncio
async def test_build_prompt_no_memory(client):
    ac, _ = client
    resp = await ac.post("/sessions/empty/prompt", json={"base_prompt": "Hello"})
    assert resp.status_code == 200
    assert resp.json()["prompt"] == "Hello"


@pytest.mark.asyncio
async def test_get_stats(client):
    ac, manager = client
    r = MemoryRecord(session_id="s1", content="fact", memory_type=MemoryType.EPISODIC)
    await manager._backend.save(r)

    resp = await ac.get("/sessions/s1/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "s1"
    assert data["total_memories"] == 1


@pytest.mark.asyncio
async def test_delete_session(client):
    ac, manager = client
    for i in range(3):
        r = MemoryRecord(session_id="s1", content=f"fact {i}")
        await manager._backend.save(r)

    resp = await ac.delete("/sessions/s1")
    assert resp.status_code == 200
    assert resp.json()["deleted"] == 3


# ── User endpoints ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_user_profile(client):
    ac, manager = client
    r = MemoryRecord(session_id="s1", user_id="u-1", content="engineer fact")
    r.embedding = [0.5, 0.5, 0.0, 0.0]
    await manager._backend.save(r)

    resp = await ac.get("/users/u-1/profile")
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == "u-1"
    assert isinstance(data["facts"], list)


@pytest.mark.asyncio
async def test_search_cross_session(client):
    ac, manager = client
    for sid in ["s1", "s2"]:
        r = MemoryRecord(session_id=sid, user_id="u-1", content="Python engineer")
        r.embedding = [0.5, 0.5, 0.0, 0.0]
        await manager._backend.save(r)

    resp = await ac.post("/users/u-1/search", json={"query": "Python", "top_k": 5})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["records"]) == 2


@pytest.mark.asyncio
async def test_delete_user(client):
    ac, manager = client
    for i in range(3):
        r = MemoryRecord(session_id=f"s{i}", user_id="u-del", content=f"fact {i}")
        await manager._backend.save(r)

    resp = await ac.delete("/users/u-del")
    assert resp.status_code == 200
    assert resp.json()["deleted"] == 3
