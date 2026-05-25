"""Unit tests for ReflectionStrategy."""
import json
import pytest
from unittest.mock import AsyncMock

from agent_memory_manager.backends.in_memory import InMemoryBackend
from agent_memory_manager.models import Message, MemoryRecord, MemoryType, Role
from agent_memory_manager.strategies.reflection import ReflectionStrategy
from agent_memory_manager.strategies.sliding_window import SlidingWindowStrategy


def _mock_embedder():
    e = AsyncMock()
    e.embed = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4])
    e.embed_batch = AsyncMock(return_value=[[0.1, 0.2, 0.3, 0.4]])
    e.dimensions = 4
    return e


def _mock_llm(insights: list[dict] | None = None):
    llm = AsyncMock()
    payload = insights or [
        {"insight": "User prefers Python.", "evidence_indices": [0], "importance": 8}
    ]
    llm.generate = AsyncMock(return_value=json.dumps(payload))
    return llm


def _msgs(*contents: str) -> list[Message]:
    return [Message(role=Role.USER, content=c) for c in contents]


@pytest.mark.asyncio
async def test_accumulator_does_not_trigger_below_threshold():
    """Reflection should NOT fire when accumulated importance < threshold."""
    backend = InMemoryBackend()
    strategy = ReflectionStrategy(
        reflection_threshold=100.0,
        delegate=SlidingWindowStrategy(window_size=10),
    )
    embedder = _mock_embedder()
    llm = _mock_llm()

    await strategy.process(_msgs("Hi"), "s1", backend, embedder, llm)
    assert strategy.get_accumulator("s1") > 0
    # Should NOT have called llm.generate for reflection yet (threshold not hit)
    # SlidingWindow adds records with importance=5; 5 < 100 threshold
    result_records = await backend.list_by_session("s1")
    reflection_records = [r for r in result_records if r.memory_type == MemoryType.REFLECTION]
    assert len(reflection_records) == 0


@pytest.mark.asyncio
async def test_reflection_triggers_above_threshold():
    """Reflection SHOULD fire once accumulated importance >= threshold."""
    backend = InMemoryBackend()
    # Low threshold to trigger immediately
    strategy = ReflectionStrategy(
        reflection_threshold=1.0,
        max_insights=2,
        delegate=SlidingWindowStrategy(window_size=10),
    )
    embedder = _mock_embedder()
    llm = _mock_llm([
        {"insight": "User works in AI.", "evidence_indices": [0], "importance": 9},
    ])

    result = await strategy.process(_msgs("I'm an AI engineer"), "s1", backend, embedder, llm)
    assert result.reflected is True
    reflection_records = [r for r in result.added if r.memory_type == MemoryType.REFLECTION]
    assert len(reflection_records) == 1
    assert "[Reflection]" in reflection_records[0].content


@pytest.mark.asyncio
async def test_accumulator_resets_after_reflection():
    backend = InMemoryBackend()
    strategy = ReflectionStrategy(
        reflection_threshold=1.0,
        delegate=SlidingWindowStrategy(window_size=10),
    )
    await strategy.process(_msgs("test"), "s1", backend, _mock_embedder(), _mock_llm())
    # After reflection fires, accumulator should be reset to 0
    assert strategy.get_accumulator("s1") == 0.0


@pytest.mark.asyncio
async def test_reflection_without_delegate():
    """ReflectionStrategy should work standalone (no inner delegate)."""
    backend = InMemoryBackend()
    # Seed some memories manually
    for i in range(3):
        r = MemoryRecord(session_id="s1", content=f"fact {i}", importance_score=60.0)
        r.embedding = [0.1, 0.2, 0.3, 0.4]
        await backend.save(r)

    strategy = ReflectionStrategy(reflection_threshold=1.0, delegate=None)
    llm = _mock_llm([
        {"insight": "User has 3 facts.", "evidence_indices": [0, 1, 2], "importance": 8}
    ])

    # Manually bump the accumulator
    strategy._accumulators["s1"] = 200.0

    # Process with empty messages (just to trigger reflection check)
    result = await strategy.process([], "s1", backend, _mock_embedder(), llm)
    assert result.reflected is True


@pytest.mark.asyncio
async def test_reflection_llm_failure_graceful():
    """If LLM fails during synthesis, reflection should not crash."""
    backend = InMemoryBackend()
    llm = AsyncMock()
    llm.generate = AsyncMock(side_effect=RuntimeError("API error"))

    strategy = ReflectionStrategy(reflection_threshold=1.0)
    strategy._accumulators["s1"] = 999.0

    r = MemoryRecord(session_id="s1", content="some memory", importance_score=5.0)
    await backend.save(r)

    result = await strategy.process([], "s1", backend, _mock_embedder(), llm)
    # Should return gracefully with no reflection records
    assert result.reflected is False
    assert len(result.added) == 0


@pytest.mark.asyncio
async def test_build_context_delegates_to_inner():
    """build_context should delegate to the inner strategy."""
    inner = SlidingWindowStrategy(window_size=5)
    strategy = ReflectionStrategy(delegate=inner)
    backend = InMemoryBackend()
    embedder = _mock_embedder()

    # Add some data via inner strategy
    await inner.process(_msgs("remember this"), "s1", backend, embedder, AsyncMock())

    context = await strategy.build_context("query", "s1", backend, embedder, token_budget=200)
    assert "remember this" in context
