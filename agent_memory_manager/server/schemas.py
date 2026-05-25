"""Pydantic request/response schemas for the REST API."""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel


# ── Request models ────────────────────────────────────────────────────────────

class MessageIn(BaseModel):
    role: str       # "user" | "assistant" | "system"
    content: str


class AddRequest(BaseModel):
    messages: list[MessageIn]
    user_id: Optional[str] = None
    metadata: Optional[dict] = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    user_id: Optional[str] = None


class PromptRequest(BaseModel):
    base_prompt: str
    token_budget: Optional[int] = None


class CrossSessionSearchRequest(BaseModel):
    query: str
    top_k: int = 10


# ── Response models ───────────────────────────────────────────────────────────

class MemoryRecordOut(BaseModel):
    id: str
    session_id: str
    user_id: Optional[str]
    memory_type: str
    content: str
    importance_score: float
    keywords: list[str]


class AddResponse(BaseModel):
    added: list[MemoryRecordOut]
    updated: list[MemoryRecordOut]
    deleted: list[MemoryRecordOut]
    entities_extracted: int
    relations_extracted: int


class SearchResponse(BaseModel):
    records: list[MemoryRecordOut]
    scores: list[float]


class PromptResponse(BaseModel):
    prompt: str


class StatsResponse(BaseModel):
    session_id: str
    total_memories: int
    episodic_count: int
    reflection_count: int
    semantic_count: int
    estimated_tokens: int
    graph_entity_count: int
    graph_relation_count: int


class UserProfileResponse(BaseModel):
    user_id: str
    facts: list[str]
    preferences: dict
    session_ids: list[str]
    total_memories: int
    raw_summary: str


class DeleteResponse(BaseModel):
    deleted: int


class NeighbourOut(BaseModel):
    relation: str
    entity: Optional[str]
    confidence: float
    distance: int


class GraphQueryResponse(BaseModel):
    entity_name: str
    neighbours: list[NeighbourOut]
    total_entities: int
    total_relations: int


def _record_out(r) -> MemoryRecordOut:
    return MemoryRecordOut(
        id=r.id,
        session_id=r.session_id,
        user_id=r.user_id,
        memory_type=r.memory_type.value,
        content=r.content,
        importance_score=r.importance_score,
        keywords=r.keywords,
    )
