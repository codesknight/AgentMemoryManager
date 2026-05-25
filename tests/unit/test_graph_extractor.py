"""Unit tests for GraphExtractor."""
import json
import pytest
from unittest.mock import AsyncMock

from agent_memory_manager.memory.graph_extractor import GraphExtractor
from agent_memory_manager.memory.semantic_memory import SemanticMemory
from agent_memory_manager.models import Message, Role


def _msgs(*contents):
    return [Message(role=Role.USER, content=c) for c in contents]


def _llm(entities=None, relations=None):
    llm = AsyncMock()
    resp = json.dumps({
        "entities": entities or [],
        "relations": relations or [],
    })
    llm.generate = AsyncMock(return_value=resp)
    return llm


@pytest.mark.asyncio
async def test_extracts_entities():
    extractor = GraphExtractor()
    graph = SemanticMemory("s1")
    llm = _llm(entities=[{"name": "Alice", "type": "person", "attributes": {"role": "engineer"}}])

    n_e, n_r = await extractor.extract(_msgs("Hi I'm Alice"), "s1", graph, llm)
    assert n_e == 1
    assert graph.entity_count == 1
    assert graph.get_entity("Alice") is not None


@pytest.mark.asyncio
async def test_extracts_relations():
    extractor = GraphExtractor()
    graph = SemanticMemory("s1")
    llm = _llm(
        entities=[
            {"name": "Alice", "type": "person", "attributes": {}},
            {"name": "DataCo", "type": "organization", "attributes": {}},
        ],
        relations=[{"subject": "Alice", "predicate": "works_at", "object": "DataCo", "confidence": 0.95}],
    )

    n_e, n_r = await extractor.extract(_msgs("Alice works at DataCo"), "s1", graph, llm)
    assert n_e == 2
    assert n_r == 1
    assert graph.relation_count == 1


@pytest.mark.asyncio
async def test_merges_existing_entity_attributes():
    extractor = GraphExtractor()
    graph = SemanticMemory("s1")

    # First pass: add Alice with role
    llm1 = _llm(entities=[{"name": "Alice", "type": "person", "attributes": {"role": "engineer"}}])
    await extractor.extract(_msgs("Alice is an engineer"), "s1", graph, llm1)

    # Second pass: add Alice with new attribute
    llm2 = _llm(entities=[{"name": "Alice", "type": "person", "attributes": {"company": "DataCo"}}])
    await extractor.extract(_msgs("Alice works at DataCo"), "s1", graph, llm2)

    alice = graph.get_entity("Alice")
    assert alice is not None
    assert alice.attributes.get("role") == "engineer"
    assert alice.attributes.get("company") == "DataCo"
    assert graph.entity_count == 1  # Not duplicated


@pytest.mark.asyncio
async def test_skips_duplicate_relations():
    extractor = GraphExtractor()
    graph = SemanticMemory("s1")
    payload = {
        "entities": [
            {"name": "Alice", "type": "person", "attributes": {}},
            {"name": "DataCo", "type": "organization", "attributes": {}},
        ],
        "relations": [{"subject": "Alice", "predicate": "works_at", "object": "DataCo", "confidence": 0.9}],
    }
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=json.dumps(payload))

    await extractor.extract(_msgs("Alice works at DataCo"), "s1", graph, llm)
    await extractor.extract(_msgs("Alice still works at DataCo"), "s1", graph, llm)

    assert graph.relation_count == 1  # Not duplicated


@pytest.mark.asyncio
async def test_llm_failure_returns_zeros():
    extractor = GraphExtractor()
    graph = SemanticMemory("s1")
    llm = AsyncMock()
    llm.generate = AsyncMock(side_effect=RuntimeError("API error"))

    n_e, n_r = await extractor.extract(_msgs("some message"), "s1", graph, llm)
    assert n_e == 0
    assert n_r == 0
    assert graph.entity_count == 0


@pytest.mark.asyncio
async def test_empty_messages_returns_zeros():
    extractor = GraphExtractor()
    graph = SemanticMemory("s1")
    llm = _llm()

    n_e, n_r = await extractor.extract([], "s1", graph, llm)
    assert n_e == 0
    assert n_r == 0


@pytest.mark.asyncio
async def test_skips_entity_with_empty_name():
    extractor = GraphExtractor()
    graph = SemanticMemory("s1")
    llm = _llm(entities=[{"name": "", "type": "person", "attributes": {}}])

    n_e, _ = await extractor.extract(_msgs("test"), "s1", graph, llm)
    assert n_e == 0
    assert graph.entity_count == 0
