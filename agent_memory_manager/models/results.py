from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .memory_record import MemoryRecord


@dataclass
class AddResult:
    added: list[MemoryRecord] = field(default_factory=list)
    updated: list[MemoryRecord] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)  # deleted memory IDs
    compressed: bool = False
    reflected: bool = False
    entities_extracted: int = 0   # entities added/updated in knowledge graph
    relations_extracted: int = 0  # relations added to knowledge graph

    @property
    def total_changes(self) -> int:
        return len(self.added) + len(self.updated) + len(self.deleted)


@dataclass
class SearchResult:
    records: list[MemoryRecord] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)

    def __iter__(self):
        return iter(zip(self.records, self.scores))


@dataclass
class ContextResult:
    context: str = ""
    token_count: int = 0
    source_memory_ids: list[str] = field(default_factory=list)


@dataclass
class CompressionResult:
    original_token_count: int = 0
    compressed_token_count: int = 0
    memories_deleted: int = 0
    summaries_created: int = 0

    @property
    def compression_ratio(self) -> float:
        if self.original_token_count == 0:
            return 0.0
        return 1.0 - self.compressed_token_count / self.original_token_count


@dataclass
class GraphQueryResult:
    """Result of a knowledge-graph neighbourhood query."""
    entity_name: str = ""
    neighbours: list[dict] = field(default_factory=list)  # {entity, relation, confidence, distance}
    total_entities: int = 0
    total_relations: int = 0


@dataclass
class MemoryStats:
    session_id: str = ""
    total_memories: int = 0
    episodic_count: int = 0
    semantic_count: int = 0
    reflection_count: int = 0
    procedural_count: int = 0
    estimated_tokens: int = 0
    oldest_memory_age_hours: Optional[float] = None
    avg_importance_score: float = 0.0
    graph_entity_count: int = 0
    graph_relation_count: int = 0
