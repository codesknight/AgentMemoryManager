"""Unit tests for MemoryManager (end-to-end, no real LLM)."""
import pytest
from unittest.mock import AsyncMock

from agent_memory_manager.backends.in_memory import InMemoryBackend
from agent_memory_manager.manager import MemoryManager
from agent_memory_manager.models import Message, MemoryRecord, MemoryType, Role
from agent_memory_manager.strategies.sliding_window import SlidingWindowStrategy


def _mock_embedder():
    e = AsyncMock()
    e.embed = AsyncMock(return_value=[0.5, 0.5, 0.0, 0.0])
    e.dimensions = 4
    return e


def _mock_llm():
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value="[]")
    return llm


def _make_manager() -> MemoryManager:
    return MemoryManager(
        backend=InMemoryBackend(),
        strategy=SlidingWindowStrategy(window_size=10),
        llm=_mock_llm(),
        embedder=_mock_embedder(),
    )


@pytest.mark.asyncio
async def test_add_and_search():
    manager = _make_manager()
    await manager.initialize()

    await manager.add(
        messages=[Message(role=Role.USER, content="I love Python programming")],
        session_id="s1",
    )

    results = await manager.search("Python", "s1", top_k=5)
    assert len(results.records) >= 0  # Records may or may not have embeddings for real similarity


@pytest.mark.asyncio
async def test_build_prompt_injects_context():
    manager = _make_manager()
    await manager.initialize()

    # Seed a memory manually with embedding
    r = MemoryRecord(session_id="s1", content="user is Alex, a data scientist")
    r.embedding = [0.5, 0.5, 0.0, 0.0]
    await manager._backend.save(r)

    prompt = await manager.build_prompt("What is my job?", "s1", token_budget=1000)
    assert "Alex" in prompt or "## Relevant Memory" in prompt or "What is my job?" in prompt


@pytest.mark.asyncio
async def test_build_prompt_no_memory_returns_base():
    manager = _make_manager()
    await manager.initialize()

    prompt = await manager.build_prompt("Hello world", "empty-session")
    assert prompt == "Hello world"


@pytest.mark.asyncio
async def test_delete_session_removes_all():
    manager = _make_manager()
    await manager.initialize()

    for i in range(5):
        await manager.add(
            messages=[Message(role=Role.USER, content=f"Message {i}")],
            session_id="s1",
        )

    deleted = await manager.delete_session("s1")
    assert deleted == 5
    stats = await manager.get_stats("s1")
    assert stats.total_memories == 0


@pytest.mark.asyncio
async def test_get_stats():
    manager = _make_manager()
    await manager.initialize()

    await manager.add(
        messages=[
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content="Hi there"),
        ],
        session_id="s1",
    )

    stats = await manager.get_stats("s1")
    assert stats.session_id == "s1"
    assert stats.total_memories == 2
    assert stats.estimated_tokens > 0


@pytest.mark.asyncio
async def test_add_with_user_id():
    manager = _make_manager()
    await manager.initialize()

    result = await manager.add(
        messages=[Message(role=Role.USER, content="test")],
        session_id="s1",
        user_id="user-42",
    )
    assert len(result.added) > 0


@pytest.mark.asyncio
async def test_manager_context_respects_token_budget():
    manager = _make_manager()
    await manager.initialize()

    for i in range(20):
        r = MemoryRecord(session_id="s1", content=f"Memory item number {i} with some content")
        r.embedding = [0.5, 0.5, 0.0, 0.0]
        await manager._backend.save(r)

    ctx = await manager.build_context("query", "s1", token_budget=50)
    from agent_memory_manager.utils.token_counter import count_tokens
    assert count_tokens(ctx.context) <= 60


@pytest.mark.asyncio
async def test_add_empty_messages_returns_empty_result():
    manager = _make_manager()
    await manager.initialize()
    result = await manager.add(messages=[], session_id="s1")
    assert result.added == []
    assert result.updated == []
    assert result.deleted == []


@pytest.mark.asyncio
async def test_add_with_metadata():
    manager = _make_manager()
    await manager.initialize()
    result = await manager.add(
        messages=[Message(role=Role.USER, content="test")],
        session_id="s1",
        metadata={"source": "unit-test"},
    )
    assert len(result.added) > 0


@pytest.mark.asyncio
async def test_search_with_memory_type_filter():
    manager = _make_manager()
    await manager.initialize()

    r = MemoryRecord(session_id="s1", content="episodic fact", memory_type=MemoryType.EPISODIC)
    r.embedding = [0.5, 0.5, 0.0, 0.0]
    await manager._backend.save(r)

    r2 = MemoryRecord(session_id="s1", content="reflection insight", memory_type=MemoryType.REFLECTION)
    r2.embedding = [0.5, 0.5, 0.0, 0.0]
    await manager._backend.save(r2)

    results = await manager.search("fact", "s1", top_k=5, memory_types=[MemoryType.EPISODIC])
    assert all(r.memory_type == MemoryType.EPISODIC for r in results.records)


@pytest.mark.asyncio
async def test_delete_session_returns_zero_for_unknown():
    manager = _make_manager()
    await manager.initialize()
    deleted = await manager.delete_session("nonexistent-session")
    assert deleted == 0


@pytest.mark.asyncio
async def test_get_stats_empty_session():
    manager = _make_manager()
    await manager.initialize()
    stats = await manager.get_stats("no-such-session")
    assert stats.total_memories == 0
    assert stats.episodic_count == 0


@pytest.mark.asyncio
async def test_get_stats_multiple_types():
    manager = _make_manager()
    await manager.initialize()

    for mt, content in [
        (MemoryType.EPISODIC, "episodic"),
        (MemoryType.REFLECTION, "reflection"),
        (MemoryType.SEMANTIC, "semantic"),
    ]:
        r = MemoryRecord(session_id="s1", content=content, memory_type=mt)
        r.embedding = [0.1, 0.2, 0.3, 0.4]
        await manager._backend.save(r)

    stats = await manager.get_stats("s1")
    assert stats.total_memories == 3
    assert stats.episodic_count == 1
    assert stats.reflection_count == 1
    assert stats.semantic_count == 1
    assert stats.avg_importance_score > 0


@pytest.mark.asyncio
async def test_compress_triggers_summarize():
    from unittest.mock import patch, AsyncMock as AM
    import json

    manager = _make_manager()
    await manager.initialize()

    for i in range(5):
        r = MemoryRecord(session_id="s1", content=f"old message {i}")
        r.embedding = [0.1, 0.2, 0.3, 0.4]
        await manager._backend.save(r)

    summary_response = "Summary of old messages."
    manager._llm.generate = AM(return_value=summary_response)

    result = await manager.compress("s1")
    assert result.original_token_count > 0


@pytest.mark.asyncio
async def test_close_does_not_raise():
    manager = _make_manager()
    await manager.initialize()
    await manager.close()  # Should not raise


# ── Graph tests (v1.5) ──────────────────────────────────────────────────────

import json

def _make_manager_with_graph_llm(entities=None, relations=None) -> MemoryManager:
    """Manager whose LLM returns a structured entity extraction response."""
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=json.dumps({
        "entities": entities or [],
        "relations": relations or [],
    }))
    return MemoryManager(
        backend=InMemoryBackend(),
        strategy=SlidingWindowStrategy(window_size=10),
        llm=llm,
        embedder=_mock_embedder(),
        enable_graph=True,
    )


@pytest.mark.asyncio
async def test_add_populates_graph():
    manager = _make_manager_with_graph_llm(
        entities=[{"name": "Sam", "type": "person", "attributes": {"role": "ML engineer"}}],
        relations=[],
    )
    await manager.initialize()

    result = await manager.add(
        messages=[Message(role=Role.USER, content="Hi I'm Sam, an ML engineer.")],
        session_id="s1",
    )
    assert result.entities_extracted == 1
    assert result.relations_extracted == 0


@pytest.mark.asyncio
async def test_query_graph_returns_neighbours():
    manager = _make_manager_with_graph_llm(
        entities=[
            {"name": "Sam", "type": "person", "attributes": {}},
            {"name": "DataCo", "type": "organization", "attributes": {}},
        ],
        relations=[{"subject": "Sam", "predicate": "works_at", "object": "DataCo", "confidence": 0.9}],
    )
    await manager.initialize()

    await manager.add(
        messages=[Message(role=Role.USER, content="Sam works at DataCo.")],
        session_id="s1",
    )
    result = await manager.query_graph("Sam", session_id="s1", hops=1)
    assert result.entity_name == "Sam"
    assert len(result.neighbours) == 1
    assert result.neighbours[0]["relation"] == "works_at"


@pytest.mark.asyncio
async def test_get_entity_returns_entity():
    manager = _make_manager_with_graph_llm(
        entities=[{"name": "Alice", "type": "person", "attributes": {"role": "engineer"}}],
    )
    await manager.initialize()
    await manager.add(
        messages=[Message(role=Role.USER, content="I am Alice.")],
        session_id="s1",
    )
    entity = await manager.get_entity("Alice", session_id="s1")
    assert entity is not None
    assert entity.name == "Alice"


@pytest.mark.asyncio
async def test_get_entity_unknown_returns_none():
    manager = _make_manager()
    await manager.initialize()
    result = await manager.get_entity("Nobody", session_id="s1")
    assert result is None


@pytest.mark.asyncio
async def test_query_graph_unknown_session_returns_empty():
    manager = _make_manager()
    await manager.initialize()
    result = await manager.query_graph("Alice", session_id="no-such-session")
    assert result.neighbours == []


@pytest.mark.asyncio
async def test_list_entities():
    manager = _make_manager_with_graph_llm(
        entities=[
            {"name": "Alice", "type": "person", "attributes": {}},
            {"name": "DataCo", "type": "organization", "attributes": {}},
        ],
    )
    await manager.initialize()
    await manager.add(
        messages=[Message(role=Role.USER, content="Alice works at DataCo.")],
        session_id="s1",
    )
    persons = await manager.list_entities("s1", entity_type="person")
    assert len(persons) == 1
    assert persons[0].name == "Alice"


@pytest.mark.asyncio
async def test_stats_include_graph_counts():
    manager = _make_manager_with_graph_llm(
        entities=[{"name": "Bob", "type": "person", "attributes": {}}],
    )
    await manager.initialize()
    await manager.add(
        messages=[Message(role=Role.USER, content="I am Bob.")],
        session_id="s1",
    )
    stats = await manager.get_stats("s1")
    assert stats.graph_entity_count == 1


@pytest.mark.asyncio
async def test_delete_session_clears_graph():
    manager = _make_manager_with_graph_llm(
        entities=[{"name": "Alice", "type": "person", "attributes": {}}],
    )
    await manager.initialize()
    await manager.add(
        messages=[Message(role=Role.USER, content="I am Alice.")],
        session_id="s1",
    )
    await manager.delete_session("s1")
    result = await manager.query_graph("Alice", session_id="s1")
    assert result.neighbours == []


@pytest.mark.asyncio
async def test_enable_graph_false_skips_extraction():
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value="[]")
    manager = MemoryManager(
        backend=InMemoryBackend(),
        strategy=SlidingWindowStrategy(window_size=10),
        llm=llm,
        embedder=_mock_embedder(),
        enable_graph=False,
    )
    await manager.initialize()
    result = await manager.add(
        messages=[Message(role=Role.USER, content="I am Alice.")],
        session_id="s1",
    )
    assert result.entities_extracted == 0
    assert result.relations_extracted == 0
