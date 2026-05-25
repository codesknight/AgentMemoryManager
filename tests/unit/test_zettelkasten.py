"""Unit tests for ZettelkastenStrategy."""
import json
import pytest
from unittest.mock import AsyncMock

from agent_memory_manager.backends.in_memory import InMemoryBackend
from agent_memory_manager.models import Message, MemoryRecord, MemoryType, Role
from agent_memory_manager.strategies.zettelkasten import ZettelkastenStrategy


def _embedder(fixed_vec: list[float] | None = None):
    e = AsyncMock()
    e.embed = AsyncMock(return_value=fixed_vec or [0.5, 0.5, 0.0, 0.0])
    e.dimensions = 4
    return e


def _llm(note_content: str = "User is a Python engineer.", keywords: list | None = None, importance: str = "7"):
    llm = AsyncMock()
    note_response = json.dumps({
        "content": note_content,
        "keywords": keywords or ["python", "engineering"],
        "context": "user background info",
    })
    # First call returns the note, second returns importance score
    llm.generate = AsyncMock(side_effect=[note_response, importance])
    return llm


def _msgs(*contents: str) -> list[Message]:
    return [Message(role=Role.USER, content=c) for c in contents]


@pytest.mark.asyncio
async def test_creates_note_and_saves():
    backend = InMemoryBackend()
    strategy = ZettelkastenStrategy(link_threshold=0.99)  # High threshold → no links yet
    llm = _llm("User likes functional programming.", ["functional", "programming"])

    result = await strategy.process(_msgs("I love functional programming"), "s1", backend, _embedder(), llm)
    assert len(result.added) == 1
    record = result.added[0]
    assert "functional programming" in record.content.lower()
    assert "functional" in record.keywords or "programming" in record.keywords


@pytest.mark.asyncio
async def test_keywords_stored_on_record():
    backend = InMemoryBackend()
    strategy = ZettelkastenStrategy()
    llm = _llm("User is a data scientist.", ["data", "science", "ml"])

    result = await strategy.process(_msgs("I'm a data scientist"), "s1", backend, _embedder(), llm)
    assert len(result.added) == 1
    assert len(result.added[0].keywords) > 0


@pytest.mark.asyncio
async def test_creates_bidirectional_links():
    """Two notes with identical embeddings should link to each other."""
    backend = InMemoryBackend()
    same_vec = [1.0, 0.0, 0.0, 0.0]
    # Low threshold so they link
    strategy = ZettelkastenStrategy(link_threshold=0.5, max_links_per_note=5)

    # First note
    llm1 = _llm("User works at TechCorp.", ["techcorp"])
    e1 = AsyncMock()
    e1.embed = AsyncMock(return_value=same_vec)
    e1.dimensions = 4
    result1 = await strategy.process(_msgs("I work at TechCorp"), "s1", backend, e1, llm1)
    first_id = result1.added[0].id

    # Second note (same embedding → high similarity → should link)
    llm2 = _llm("User is a backend engineer at TechCorp.", ["techcorp", "backend"])
    e2 = AsyncMock()
    e2.embed = AsyncMock(return_value=same_vec)
    e2.dimensions = 4
    result2 = await strategy.process(_msgs("I'm a backend engineer"), "s1", backend, e2, llm2)
    second_record = result2.added[0]

    # Second note should link to first
    assert first_id in second_record.links

    # First note should now have a backlink to second (bidirectional)
    first_record = await backend.get(first_id)
    assert first_record is not None
    assert second_record.id in first_record.links


@pytest.mark.asyncio
async def test_build_context_includes_linked_notes():
    """build_context should follow links and include linked notes in context."""
    backend = InMemoryBackend()
    strategy = ZettelkastenStrategy(link_hops=1)

    # Seed two notes with explicit link
    vec_a = [1.0, 0.0]
    vec_b = [0.0, 1.0]

    note_a = MemoryRecord(session_id="s1", content="User's name is Alex.")
    note_a.embedding = vec_a
    note_a.keywords = ["name", "alex"]
    await backend.save(note_a)

    note_b = MemoryRecord(session_id="s1", content="Alex works at DataCo.")
    note_b.embedding = vec_b
    note_b.links = [note_a.id]
    note_b.keywords = ["work", "dataco"]
    await backend.save(note_b)

    # Query matches note_b directly; note_a is reached via link
    embedder = AsyncMock()
    embedder.embed = AsyncMock(return_value=vec_b)
    embedder.dimensions = 2

    context = await strategy.build_context("where does alex work", "s1", backend, embedder, token_budget=500)

    assert "Alex works at DataCo." in context
    assert "User's name is Alex." in context  # reached via link-hop


@pytest.mark.asyncio
async def test_llm_failure_falls_back_gracefully():
    """If LLM fails, strategy should fall back to raw content."""
    backend = InMemoryBackend()
    strategy = ZettelkastenStrategy()

    llm = AsyncMock()
    llm.generate = AsyncMock(side_effect=RuntimeError("API down"))

    result = await strategy.process(_msgs("some message"), "s1", backend, _embedder(), llm)
    # Fallback: raw content is stored
    assert len(result.added) == 1
    assert "some message" in result.added[0].content


@pytest.mark.asyncio
async def test_empty_messages_returns_empty_result():
    backend = InMemoryBackend()
    strategy = ZettelkastenStrategy()
    result = await strategy.process([], "s1", backend, _embedder(), AsyncMock())
    assert result.added == []


@pytest.mark.asyncio
async def test_importance_score_clamped():
    """Even if LLM returns out-of-range importance, it should be clamped to 1–10."""
    backend = InMemoryBackend()
    strategy = ZettelkastenStrategy()
    llm = AsyncMock()
    note_resp = json.dumps({"content": "test fact", "keywords": [], "context": ""})
    llm.generate = AsyncMock(side_effect=[note_resp, "99"])  # invalid importance

    result = await strategy.process(_msgs("test"), "s1", backend, _embedder(), llm)
    assert len(result.added) == 1
    assert result.added[0].importance_score <= 10.0
