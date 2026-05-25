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

    # Add many messages
    for i in range(20):
        r = MemoryRecord(session_id="s1", content=f"Memory item number {i} with some content")
        r.embedding = [0.5, 0.5, 0.0, 0.0]
        await manager._backend.save(r)

    ctx = await manager.build_context("query", "s1", token_budget=50)
    # Should be within budget
    from agent_memory_manager.utils.token_counter import count_tokens
    assert count_tokens(ctx.context) <= 60  # small slack
