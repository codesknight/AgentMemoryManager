"""Unit tests for MemoryManager cross-session user memory (v2.0-A)."""
import json
import pytest
from unittest.mock import AsyncMock

from agent_memory_manager.backends.in_memory import InMemoryBackend
from agent_memory_manager.manager import MemoryManager
from agent_memory_manager.models import Message, MemoryRecord, Role
from agent_memory_manager.strategies.sliding_window import SlidingWindowStrategy


def _mock_embedder():
    e = AsyncMock()
    e.embed = AsyncMock(return_value=[0.5, 0.5, 0.0, 0.0])
    e.dimensions = 4
    return e


def _mock_llm(profile_response=None):
    llm = AsyncMock()
    default = json.dumps({
        "facts": ["User is a backend engineer.", "User prefers Python."],
        "preferences": {"language": "Python"},
        "raw_summary": "A backend engineer who prefers Python.",
    })
    llm.generate = AsyncMock(return_value=profile_response or default)
    return llm


def _make_manager(**kwargs) -> MemoryManager:
    return MemoryManager(
        backend=InMemoryBackend(),
        strategy=SlidingWindowStrategy(window_size=10),
        llm=_mock_llm(**kwargs),
        embedder=_mock_embedder(),
    )


# ── list_by_user / delete_by_user ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_by_user_returns_user_records():
    manager = _make_manager()
    await manager.initialize()

    for i in range(3):
        r = MemoryRecord(session_id=f"s{i}", user_id="u-1", content=f"msg {i}")
        r.embedding = [0.5, 0.5, 0.0, 0.0]
        await manager._backend.save(r)

    # Record for different user
    r2 = MemoryRecord(session_id="s99", user_id="u-2", content="other user")
    await manager._backend.save(r2)

    records = await manager._backend.list_by_user("u-1")
    assert len(records) == 3
    assert all(r.user_id == "u-1" for r in records)


@pytest.mark.asyncio
async def test_delete_by_user_removes_only_that_user():
    manager = _make_manager()
    await manager.initialize()

    for i in range(3):
        r = MemoryRecord(session_id="s1", user_id="u-1", content=f"msg {i}")
        await manager._backend.save(r)
    r2 = MemoryRecord(session_id="s2", user_id="u-2", content="other")
    await manager._backend.save(r2)

    deleted = await manager._backend.delete_by_user("u-1")
    assert deleted == 3
    remaining = await manager._backend.list_by_user("u-2")
    assert len(remaining) == 1


# ── build_user_profile ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_build_user_profile_synthesizes_facts():
    manager = _make_manager()
    await manager.initialize()

    for i in range(3):
        r = MemoryRecord(session_id=f"s{i}", user_id="u-1",
                         content=f"User fact {i}")
        r.embedding = [0.5, 0.5, 0.0, 0.0]
        await manager._backend.save(r)

    profile = await manager.build_user_profile("u-1")
    assert profile.user_id == "u-1"
    assert profile.total_memories == 3
    assert len(profile.facts) > 0
    assert "Python" in profile.preferences.get("language", "")


@pytest.mark.asyncio
async def test_build_user_profile_cached_on_second_call():
    manager = _make_manager()
    await manager.initialize()

    r = MemoryRecord(session_id="s1", user_id="u-1", content="fact")
    await manager._backend.save(r)

    p1 = await manager.build_user_profile("u-1")
    # Mutate cache to verify second call returns cached
    p1.raw_summary = "CACHED"
    manager._user_profiles["u-1"] = p1

    p2 = await manager.build_user_profile("u-1")
    assert p2.raw_summary == "CACHED"


@pytest.mark.asyncio
async def test_build_user_profile_force_rebuild():
    manager = _make_manager()
    await manager.initialize()

    r = MemoryRecord(session_id="s1", user_id="u-1", content="fact")
    await manager._backend.save(r)

    p1 = await manager.build_user_profile("u-1")
    p1.raw_summary = "STALE"
    manager._user_profiles["u-1"] = p1

    p2 = await manager.build_user_profile("u-1", force_rebuild=True)
    assert p2.raw_summary != "STALE"


@pytest.mark.asyncio
async def test_build_user_profile_empty_user_returns_profile():
    manager = _make_manager()
    await manager.initialize()
    profile = await manager.build_user_profile("no-memories-user")
    assert profile.user_id == "no-memories-user"
    assert profile.total_memories == 0


# ── get_user_profile ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_user_profile_none_before_build():
    manager = _make_manager()
    await manager.initialize()
    result = await manager.get_user_profile("u-ghost")
    assert result is None


@pytest.mark.asyncio
async def test_get_user_profile_returns_after_build():
    manager = _make_manager()
    await manager.initialize()
    r = MemoryRecord(session_id="s1", user_id="u-1", content="fact")
    await manager._backend.save(r)
    await manager.build_user_profile("u-1")
    profile = await manager.get_user_profile("u-1")
    assert profile is not None
    assert profile.user_id == "u-1"


# ── search_cross_session ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_search_cross_session_finds_user_records():
    manager = _make_manager()
    await manager.initialize()

    for i, sid in enumerate(["s1", "s2", "s3"]):
        r = MemoryRecord(session_id=sid, user_id="u-1",
                         content=f"Python fact {i}")
        r.embedding = [0.5, 0.5, 0.0, 0.0]
        await manager._backend.save(r)

    # Different user
    r_other = MemoryRecord(session_id="s4", user_id="u-2", content="other user fact")
    r_other.embedding = [0.5, 0.5, 0.0, 0.0]
    await manager._backend.save(r_other)

    results = await manager.search_cross_session("u-1", "Python", top_k=10)
    assert all(r.user_id == "u-1" for r in results.records)
    assert len(results.records) == 3


# ── delete_user ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delete_user_removes_memories_and_profile():
    manager = _make_manager()
    await manager.initialize()

    for i in range(4):
        r = MemoryRecord(session_id=f"s{i}", user_id="u-1", content=f"fact {i}")
        await manager._backend.save(r)

    await manager.build_user_profile("u-1")
    assert await manager.get_user_profile("u-1") is not None

    deleted = await manager.delete_user("u-1")
    assert deleted == 4
    assert await manager.get_user_profile("u-1") is None
    remaining = await manager._backend.list_by_user("u-1")
    assert remaining == []
