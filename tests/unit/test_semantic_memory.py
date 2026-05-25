"""Unit tests for SemanticMemory (NetworkX knowledge graph)."""
import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta, timezone

networkx = pytest.importorskip("networkx", reason="networkx not installed")

from agent_memory_manager.memory.semantic_memory import SemanticMemory
from agent_memory_manager.models.entity import Entity, Relation


def _entity(name: str, etype: str = "person", **attrs) -> Entity:
    return Entity(session_id="s1", name=name, entity_type=etype, attributes=attrs)


def _relation(subj: str, pred: str, obj: str, confidence: float = 1.0) -> Relation:
    return Relation(
        session_id="s1",
        subject_id=subj,
        predicate=pred,
        object_id=obj,
        confidence=confidence,
    )


def test_add_and_get_entity():
    sm = SemanticMemory("s1")
    sm.add_entity(_entity("Alice", role="engineer"))
    e = sm.get_entity("Alice")
    assert e is not None
    assert e.name == "Alice"
    assert e.attributes["role"] == "engineer"


def test_entity_name_case_insensitive():
    sm = SemanticMemory("s1")
    sm.add_entity(_entity("Alice"))
    assert sm.get_entity("alice") is not None
    assert sm.get_entity("ALICE") is not None


def test_add_relation_and_get_neighbours():
    sm = SemanticMemory("s1")
    sm.add_entity(_entity("Alice"))
    sm.add_entity(_entity("TechCorp", etype="organization"))
    sm.add_relation(_relation("Alice", "works_at", "TechCorp"))

    neighbours = sm.get_neighbours("Alice", hops=1)
    names = [n["entity"].name for n in neighbours if n["entity"]]
    assert "TechCorp" in names


def test_multi_hop_query():
    sm = SemanticMemory("s1")
    sm.add_entity(_entity("Alice"))
    sm.add_entity(_entity("TechCorp", etype="organization"))
    sm.add_entity(_entity("SiliconValley", etype="location"))
    sm.add_relation(_relation("Alice", "works_at", "TechCorp"))
    sm.add_relation(_relation("TechCorp", "located_in", "SiliconValley"))

    neighbours = sm.get_neighbours("Alice", hops=2)
    names = [n["entity"].name for n in neighbours if n["entity"]]
    assert "TechCorp" in names
    assert "SiliconValley" in names


def test_single_hop_does_not_reach_2_hops():
    sm = SemanticMemory("s1")
    sm.add_entity(_entity("A"))
    sm.add_entity(_entity("B"))
    sm.add_entity(_entity("C"))
    sm.add_relation(_relation("A", "knows", "B"))
    sm.add_relation(_relation("B", "knows", "C"))

    neighbours_1hop = sm.get_neighbours("A", hops=1)
    names = [n["entity"].name for n in neighbours_1hop if n["entity"]]
    assert "B" in names
    assert "C" not in names


def test_invalidate_relation_filters_expired():
    sm = SemanticMemory("s1")
    sm.add_entity(_entity("Alice"))
    sm.add_entity(_entity("OldCo", etype="organization"))
    sm.add_relation(_relation("Alice", "works_at", "OldCo"))

    # Get relation id
    rel_id = sm._relations[0].id
    sm.invalidate_relation(rel_id)

    # current_only=True should exclude expired relation
    neighbours = sm.get_neighbours("Alice", hops=1, current_only=True)
    names = [n["entity"].name for n in neighbours if n["entity"]]
    assert "OldCo" not in names

    # current_only=False should include it
    neighbours_all = sm.get_neighbours("Alice", hops=1, current_only=False)
    names_all = [n["entity"].name for n in neighbours_all if n["entity"]]
    assert "OldCo" in names_all


def test_search_entities_by_type():
    sm = SemanticMemory("s1")
    sm.add_entity(_entity("Alice", etype="person"))
    sm.add_entity(_entity("Bob", etype="person"))
    sm.add_entity(_entity("TechCorp", etype="organization"))

    persons = sm.search_entities(entity_type="person")
    assert len(persons) == 2
    orgs = sm.search_entities(entity_type="organization")
    assert len(orgs) == 1


def test_entity_and_relation_counts():
    sm = SemanticMemory("s1")
    sm.add_entity(_entity("A"))
    sm.add_entity(_entity("B"))
    sm.add_relation(_relation("A", "knows", "B"))
    assert sm.entity_count == 2
    assert sm.relation_count == 1


def test_serialization_roundtrip():
    sm = SemanticMemory("s1")
    sm.add_entity(_entity("Alice", role="engineer"))
    sm.add_entity(_entity("TechCorp", etype="organization"))
    sm.add_relation(_relation("Alice", "works_at", "TechCorp"))

    data = sm.to_dict()
    sm2 = SemanticMemory.from_dict(data)

    assert sm2.entity_count == 2
    assert sm2.relation_count == 1
    alice = sm2.get_entity("Alice")
    assert alice is not None
    assert alice.attributes.get("role") == "engineer"


def test_save_and_load_file():
    sm = SemanticMemory("s1")
    sm.add_entity(_entity("Alice"))
    sm.add_entity(_entity("Bob"))
    sm.add_relation(_relation("Alice", "knows", "Bob"))

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)

    sm.save(path)
    sm_loaded = SemanticMemory.load(path)
    path.unlink()

    assert sm_loaded.entity_count == 2
    assert sm_loaded.relation_count == 1


def test_unknown_entity_in_relation_skipped():
    """Adding a relation with unknown entity should not crash."""
    sm = SemanticMemory("s1")
    sm.add_entity(_entity("Alice"))
    # "UnknownCo" not added to graph
    rel = _relation("Alice", "works_at", "UnknownCo")
    sm.add_relation(rel)  # Should not raise
    assert sm.relation_count == 0  # Not stored


def test_get_current_relations():
    sm = SemanticMemory("s1")
    sm.add_entity(_entity("A"))
    sm.add_entity(_entity("B"))
    sm.add_entity(_entity("C"))
    sm.add_relation(_relation("A", "knows", "B"))
    sm.add_relation(_relation("A", "knows", "C"))

    rel_id = sm._relations[0].id
    sm.invalidate_relation(rel_id)

    current = sm.get_current_relations()
    assert len(current) == 1
