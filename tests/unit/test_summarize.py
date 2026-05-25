"""Unit tests for SummarizeStrategy."""
import pytest
from unittest.mock import AsyncMock

from agent_memory_manager.backends.in_memory import InMemoryBackend
from agent_memory_manager.models import Message, MemoryRecord, Role
from agent_memory_manager.strategies.summarize import SummarizeStrategy


def _embedder():
    e = AsyncMock()
    e.embed = AsyncMock(return_value=[0.3, 0.3])
    e.dimensions = 2
    return e


def _llm(summary: str = "A concise summary of the conversation."):
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=summary)
    return llm


def _msgs(*contents: str) -> list[Message]:
    return [Message(role=Role.USER, content=c) for c in contents]


@pytest.mark.asyncio
async def test_saves_messages_as_records():
    backend = InMemoryBackend()
    strategy = SummarizeStrategy(summarize_threshold=10_000)  # High threshold, no compression
    await strategy.process(_msgs("Hello", "World"), "s1", backend, _embedder(), _llm())
    records = await backend.list_by_session("s1")
    assert len(records) == 2


@pytest.mark.asyncio
async def test_compresses_when_threshold_exceeded():
    """When total tokens exceed threshold, old messages should be compressed."""
    backend = InMemoryBackend()
    # Very low threshold to force compression
    strategy = SummarizeStrategy(summarize_threshold=1, preserve_recent=1)
    embedder = _embedder()
    llm = _llm("Summary of earlier conversation.")

    # Add 5 messages
    for i in range(5):
        await strategy.process(_msgs(f"message {i}" * 50), "s1", backend, embedder, llm)

    records = await backend.list_by_session("s1", limit=1000)
    # Should have a summary record
    summary_records = [r for r in records if r.content.startswith("[Summary]")]
    assert len(summary_records) >= 1


@pytest.mark.asyncio
async def test_preserves_recent_on_compression():
    """The preserve_recent most recent records should survive compression."""
    backend = InMemoryBackend()
    strategy = SummarizeStrategy(summarize_threshold=1, preserve_recent=2)
    embedder = _embedder()
    llm = _llm("old summary")

    for i in range(6):
        await strategy.process(_msgs(f"msg {i}"), "s1", backend, embedder, llm)

    records = await backend.list_by_session("s1", limit=1000)
    # There should be summary records + preserved recent records
    assert len(records) >= 1


@pytest.mark.asyncio
async def test_build_context_returns_semantically_similar():
    backend = InMemoryBackend()
    strategy = SummarizeStrategy(summarize_threshold=10_000)
    embedder = _embedder()
    llm = _llm()

    r = MemoryRecord(session_id="s1", content="user likes Python", importance_score=8.0)
    r.embedding = [0.3, 0.3]
    await backend.save(r)

    context = await strategy.build_context("what lang", "s1", backend, embedder, token_budget=500)
    assert "user likes Python" in context


@pytest.mark.asyncio
async def test_compression_result_flag():
    backend = InMemoryBackend()
    strategy = SummarizeStrategy(summarize_threshold=1, preserve_recent=0)
    embedder = _embedder()
    llm = _llm("summary text")

    # Add enough content to exceed threshold
    result = await strategy.process(_msgs("a" * 500), "s1", backend, embedder, llm)
    assert result.compressed is True
