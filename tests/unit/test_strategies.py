"""Unit tests for memory strategies using mock LLM and embedder."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from agent_memory_manager.backends.in_memory import InMemoryBackend
from agent_memory_manager.models import Message, Role
from agent_memory_manager.strategies.sliding_window import SlidingWindowStrategy
from agent_memory_manager.strategies.pipeline import StrategyPipeline


def _mock_embedder(dim: int = 4):
    embedder = AsyncMock()
    embedder.embed = AsyncMock(return_value=[0.1] * dim)
    embedder.embed_batch = AsyncMock(return_value=[[0.1] * dim])
    embedder.dimensions = dim
    return embedder


def _mock_llm(response: str = "[]"):
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=response)
    return llm


def _msgs(*contents: str) -> list[Message]:
    return [Message(role=Role.USER, content=c) for c in contents]


@pytest.mark.asyncio
async def test_sliding_window_saves_messages():
    backend = InMemoryBackend()
    strategy = SlidingWindowStrategy(window_size=10)

    await strategy.process(_msgs("Hello", "World"), "s1", backend, _mock_embedder(), _mock_llm())

    records = await backend.list_by_session("s1")
    assert len(records) == 2
    assert "Hello" in records[0].content


@pytest.mark.asyncio
async def test_sliding_window_context_respects_budget():
    backend = InMemoryBackend()
    strategy = SlidingWindowStrategy(window_size=5)
    embedder = _mock_embedder()
    llm = _mock_llm()

    # Add many messages
    for i in range(10):
        await strategy.process(_msgs(f"Message number {i}"), "s1", backend, embedder, llm)

    context = await strategy.build_context("query", "s1", backend, embedder, token_budget=10)
    # Context should be short due to tight budget
    assert len(context) < 200


@pytest.mark.asyncio
async def test_pipeline_merges_results():
    backend = InMemoryBackend()
    s1 = SlidingWindowStrategy(window_size=10)
    s2 = SlidingWindowStrategy(window_size=10)
    pipeline = StrategyPipeline([s1, s2])
    embedder = _mock_embedder()
    llm = _mock_llm()

    result = await pipeline.process(_msgs("fact"), "s1", backend, embedder, llm)
    # Both strategies save the same message → 2 records total
    assert len(result.added) == 2
