"""Semantic memory layer backed by a NetworkX directed knowledge graph.

Stores entities and temporal relations extracted from conversations.
Supports multi-hop queries and time-aware relation filtering.

Reference: Temporal KG design inspired by Zep/Graphiti (arXiv:2501.13956).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from agent_memory_manager.models.entity import Entity, Relation

try:
    import networkx as nx
except ImportError as exc:
    raise ImportError(
        "Install 'networkx' to use SemanticMemory: pip install networkx"
    ) from exc

logger = logging.getLogger(__name__)


class SemanticMemory:
    """In-process knowledge graph for entity-relation storage.

    Features:
    - Entities (Person, Organization, Concept, …) as graph nodes
    - Typed, directed, temporally-aware relations as graph edges
    - Multi-hop neighbourhood queries
    - JSON serialization for persistence across sessions
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._graph: nx.DiGraph = nx.DiGraph()
        self._entities: dict[str, Entity] = {}   # name → Entity
        self._relations: list[Relation] = []

    # ────────── Write ──────────

    def add_entity(self, entity: Entity) -> None:
        """Add or update an entity node in the graph."""
        self._entities[entity.name.lower()] = entity
        self._graph.add_node(
            entity.name.lower(),
            id=entity.id,
            name=entity.name,
            entity_type=entity.entity_type,
            attributes=entity.attributes,
        )

    def add_relation(self, relation: Relation) -> None:
        """Add a directed, temporally-aware relation edge."""
        subj = self._resolve_name(relation.subject_id)
        obj = self._resolve_name(relation.object_id)
        if subj is None or obj is None:
            logger.debug(
                "Skipping relation %s: subject or object not in graph", relation.id
            )
            return
        self._relations.append(relation)
        self._graph.add_edge(
            subj,
            obj,
            id=relation.id,
            predicate=relation.predicate,
            confidence=relation.confidence,
            valid_from=relation.valid_from.isoformat(),
            valid_to=relation.valid_to.isoformat() if relation.valid_to else None,
        )

    def invalidate_relation(self, relation_id: str) -> bool:
        """Mark a relation as no longer valid (soft delete)."""
        now = datetime.now(timezone.utc)
        for rel in self._relations:
            if rel.id == relation_id and rel.valid_to is None:
                rel.valid_to = now
                # Update edge data in graph
                subj = self._resolve_name(rel.subject_id)
                obj = self._resolve_name(rel.object_id)
                if subj and obj and self._graph.has_edge(subj, obj):
                    self._graph[subj][obj]["valid_to"] = now.isoformat()
                return True
        return False

    # ────────── Query ──────────

    def get_entity(self, name: str) -> Optional[Entity]:
        return self._entities.get(name.lower())

    def get_neighbours(
        self,
        name: str,
        hops: int = 1,
        current_only: bool = True,
    ) -> list[dict]:
        """Return entities reachable within `hops` from `name`.

        Args:
            name: Entity name (case-insensitive).
            hops: Maximum number of hops to traverse.
            current_only: If True, skip expired relations.
        Returns:
            List of dicts with keys: entity, relation, distance.
        """
        start = name.lower()
        if start not in self._graph:
            return []

        results: list[dict] = []
        visited = {start}
        frontier = [(start, 0)]

        while frontier:
            node, depth = frontier.pop(0)
            if depth >= hops:
                continue
            for neighbour in self._graph.successors(node):
                if neighbour in visited:
                    continue
                edge_data = self._graph[node][neighbour]
                if current_only and edge_data.get("valid_to") is not None:
                    continue
                entity = self._entities.get(neighbour)
                results.append({
                    "entity": entity,
                    "relation": edge_data.get("predicate"),
                    "confidence": edge_data.get("confidence", 1.0),
                    "distance": depth + 1,
                })
                visited.add(neighbour)
                frontier.append((neighbour, depth + 1))

        return results

    def search_entities(
        self,
        entity_type: Optional[str] = None,
        attribute_key: Optional[str] = None,
        attribute_value: Optional[str] = None,
    ) -> list[Entity]:
        """Filter entities by type or attribute."""
        results = list(self._entities.values())
        if entity_type:
            results = [e for e in results if e.entity_type == entity_type]
        if attribute_key and attribute_value:
            results = [
                e for e in results
                if str(e.attributes.get(attribute_key, "")).lower()
                == attribute_value.lower()
            ]
        return results

    def get_current_relations(self) -> list[Relation]:
        """Return only relations that are currently valid."""
        return [r for r in self._relations if r.is_current]

    @property
    def entity_count(self) -> int:
        return len(self._entities)

    @property
    def relation_count(self) -> int:
        return len(self._relations)

    # ────────── Serialization ──────────

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "entities": [e.model_dump(mode="json") for e in self._entities.values()],
            "relations": [r.model_dump(mode="json") for r in self._relations],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SemanticMemory":
        sm = cls(session_id=data["session_id"])
        for e_data in data.get("entities", []):
            sm.add_entity(Entity(**e_data))
        for r_data in data.get("relations", []):
            sm.add_relation(Relation(**r_data))
        return sm

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), default=str), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SemanticMemory":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    # ────────── private ──────────

    def _resolve_name(self, entity_id_or_name: str) -> Optional[str]:
        """Resolve an entity ID or name to its graph node key (lowercase name)."""
        # Try direct name lookup first
        key = entity_id_or_name.lower()
        if key in self._entities:
            return key
        # Try lookup by ID
        for name, entity in self._entities.items():
            if entity.id == entity_id_or_name:
                return name
        return None
