"""Unit tests for GraphStore (SQLite persistence for SemanticMemory)."""
import pytest
import tempfile
from pathlib import Path

from agent_memory_manager.memory.graph_store import GraphStore
from agent_memory_manager.memory.semantic_memory import SemanticMemory
from agent_memory_manager.models.entity import Entity, Relation


def _make_graph(session_id: str) -> SemanticMemory:
    sm = SemanticMemory(session_id)
    sm.add_entity(Entity(session_id=session_id, name="Alice", entity_type="person"))
    sm.add_entity(Entity(session_id=session_id, name="DataCo", entity_type="organization"))
    sm.add_relation(Relation(
        session_id=session_id, subject_id="Alice",
        predicate="works_at", object_id="DataCo",
    ))
    return sm


@pytest.fixture
async def store():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    gs = GraphStore(path)
    await gs.initialize()
    yield gs
    await gs.close()
    Path(path).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_save_and_load(store):
    graph = _make_graph("s1")
    await store.save(graph)

    loaded = await store.load("s1")
    assert loaded is not None
    assert loaded.entity_count == 2
    assert loaded.relation_count == 1
    assert loaded.get_entity("Alice") is not None


@pytest.mark.asyncio
async def test_load_nonexistent_returns_none(store):
    result = await store.load("no-such-session")
    assert result is None


@pytest.mark.asyncio
async def test_save_overwrites_existing(store):
    graph1 = _make_graph("s1")
    await store.save(graph1)

    graph2 = SemanticMemory("s1")
    graph2.add_entity(Entity(session_id="s1", name="Bob", entity_type="person"))
    await store.save(graph2)

    loaded = await store.load("s1")
    assert loaded is not None
    assert loaded.entity_count == 1
    assert loaded.get_entity("Bob") is not None
    assert loaded.get_entity("Alice") is None


@pytest.mark.asyncio
async def test_delete(store):
    await store.save(_make_graph("s1"))
    deleted = await store.delete("s1")
    assert deleted is True
    assert await store.load("s1") is None


@pytest.mark.asyncio
async def test_delete_nonexistent_returns_false(store):
    result = await store.delete("no-such-session")
    assert result is False


@pytest.mark.asyncio
async def test_list_sessions(store):
    await store.save(_make_graph("s1"))
    await store.save(_make_graph("s2"))

    sessions = await store.list_sessions()
    assert "s1" in sessions
    assert "s2" in sessions
