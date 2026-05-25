"""Unit tests for UserProfile model and UserProfileStore."""
import json
import pytest
import tempfile
from pathlib import Path

from agent_memory_manager.models.user_profile import UserProfile
from agent_memory_manager.memory.user_profile_store import UserProfileStore


def _make_profile(user_id: str = "u-1") -> UserProfile:
    return UserProfile(
        user_id=user_id,
        facts=["User is a backend engineer.", "User prefers Python."],
        preferences={"language": "Python"},
        session_ids=["s1", "s2"],
        total_memories=10,
        raw_summary="A backend engineer who prefers Python.",
    )


# ── Model serialisation ──────────────────────────────────────────────────────

def test_user_profile_roundtrip_json():
    p = _make_profile()
    restored = UserProfile.from_json(p.to_json())
    assert restored.user_id == p.user_id
    assert restored.facts == p.facts
    assert restored.preferences == p.preferences
    assert restored.session_ids == p.session_ids
    assert restored.total_memories == p.total_memories
    assert restored.raw_summary == p.raw_summary


def test_user_profile_empty_defaults():
    p = UserProfile(user_id="u-empty")
    assert p.facts == []
    assert p.preferences == {}
    assert p.session_ids == []
    assert p.total_memories == 0


# ── UserProfileStore ─────────────────────────────────────────────────────────

@pytest.fixture
async def store():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    gs = UserProfileStore(path)
    await gs.initialize()
    yield gs
    await gs.close()
    Path(path).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_save_and_load(store):
    p = _make_profile("u-1")
    await store.save(p)
    loaded = await store.load("u-1")
    assert loaded is not None
    assert loaded.user_id == "u-1"
    assert loaded.facts == p.facts
    assert loaded.preferences == p.preferences


@pytest.mark.asyncio
async def test_load_nonexistent_returns_none(store):
    result = await store.load("nobody")
    assert result is None


@pytest.mark.asyncio
async def test_save_overwrites(store):
    await store.save(_make_profile("u-1"))
    p2 = UserProfile(user_id="u-1", facts=["Updated fact."], total_memories=99)
    await store.save(p2)
    loaded = await store.load("u-1")
    assert loaded is not None
    assert loaded.facts == ["Updated fact."]
    assert loaded.total_memories == 99


@pytest.mark.asyncio
async def test_delete(store):
    await store.save(_make_profile("u-1"))
    deleted = await store.delete("u-1")
    assert deleted is True
    assert await store.load("u-1") is None


@pytest.mark.asyncio
async def test_delete_nonexistent_returns_false(store):
    assert await store.delete("nobody") is False


@pytest.mark.asyncio
async def test_list_users(store):
    await store.save(_make_profile("u-1"))
    await store.save(_make_profile("u-2"))
    users = await store.list_users()
    assert "u-1" in users
    assert "u-2" in users
