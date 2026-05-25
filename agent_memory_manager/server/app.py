"""FastAPI application factory for AgentMemoryManager REST API (v2.0-B)."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException

from agent_memory_manager.manager import MemoryManager
from agent_memory_manager.models import Message, Role

from .schemas import (
    AddRequest, AddResponse,
    CrossSessionSearchRequest,
    DeleteResponse,
    GraphQueryResponse, NeighbourOut,
    PromptRequest, PromptResponse,
    SearchRequest, SearchResponse,
    StatsResponse,
    UserProfileResponse,
    _record_out,
)


def create_app(manager: MemoryManager) -> FastAPI:
    """Create a FastAPI app wired to the given MemoryManager.

    Usage::

        from agent_memory_manager.server import create_app
        manager = MemoryManager(...)
        app = create_app(manager)
        # uvicorn agent_memory_manager.server.app:app --reload
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await manager.initialize()
        yield
        await manager.close()

    app = FastAPI(
        title="AgentMemoryManager API",
        version="2.0.0",
        description="REST interface for LLM agent memory management.",
        lifespan=lifespan,
    )

    # ── Session routes ────────────────────────────────────────────────────────

    @app.post("/sessions/{session_id}/add", response_model=AddResponse)
    async def add_messages(session_id: str, body: AddRequest):
        """Ingest conversation messages into memory."""
        messages = [
            Message(role=Role(m.role), content=m.content)
            for m in body.messages
        ]
        result = await manager.add(
            messages=messages,
            session_id=session_id,
            user_id=body.user_id,
            metadata=body.metadata,
        )
        return AddResponse(
            added=[_record_out(r) for r in result.added],
            updated=[_record_out(r) for r in result.updated],
            deleted=[_record_out(r) for r in result.deleted],
            entities_extracted=result.entities_extracted,
            relations_extracted=result.relations_extracted,
        )

    @app.post("/sessions/{session_id}/search", response_model=SearchResponse)
    async def search_memories(session_id: str, body: SearchRequest):
        """Semantic search within a session."""
        result = await manager.search(body.query, session_id, top_k=body.top_k)
        return SearchResponse(
            records=[_record_out(r) for r in result.records],
            scores=result.scores,
        )

    @app.post("/sessions/{session_id}/prompt", response_model=PromptResponse)
    async def build_prompt(session_id: str, body: PromptRequest):
        """Return a memory-enhanced prompt."""
        prompt = await manager.build_prompt(
            body.base_prompt, session_id, token_budget=body.token_budget
        )
        return PromptResponse(prompt=prompt)

    @app.get("/sessions/{session_id}/stats", response_model=StatsResponse)
    async def get_stats(session_id: str):
        """Return memory statistics for a session."""
        s = await manager.get_stats(session_id)
        return StatsResponse(
            session_id=s.session_id,
            total_memories=s.total_memories,
            episodic_count=s.episodic_count,
            reflection_count=s.reflection_count,
            semantic_count=s.semantic_count,
            estimated_tokens=s.estimated_tokens,
            graph_entity_count=s.graph_entity_count,
            graph_relation_count=s.graph_relation_count,
        )

    @app.delete("/sessions/{session_id}", response_model=DeleteResponse)
    async def delete_session(session_id: str):
        """Delete all memories for a session (GDPR)."""
        count = await manager.delete_session(session_id)
        return DeleteResponse(deleted=count)

    # ── Graph routes ──────────────────────────────────────────────────────────

    @app.get("/sessions/{session_id}/graph/{entity_name}",
             response_model=GraphQueryResponse)
    async def query_graph(session_id: str, entity_name: str, hops: int = 1):
        """Query the knowledge graph for an entity's neighbourhood."""
        result = await manager.query_graph(entity_name, session_id=session_id, hops=hops)
        neighbours = [
            NeighbourOut(
                relation=n["relation"],
                entity=n["entity"].name if n["entity"] else None,
                confidence=n["confidence"],
                distance=n["distance"],
            )
            for n in result.neighbours
        ]
        return GraphQueryResponse(
            entity_name=result.entity_name,
            neighbours=neighbours,
            total_entities=result.total_entities,
            total_relations=result.total_relations,
        )

    # ── User routes ───────────────────────────────────────────────────────────

    @app.get("/users/{user_id}/profile", response_model=UserProfileResponse)
    async def get_user_profile(user_id: str, rebuild: bool = False):
        """Return (or synthesize) a user profile aggregated across all sessions."""
        profile = await manager.build_user_profile(user_id, force_rebuild=rebuild)
        return UserProfileResponse(
            user_id=profile.user_id,
            facts=profile.facts,
            preferences=profile.preferences,
            session_ids=profile.session_ids,
            total_memories=profile.total_memories,
            raw_summary=profile.raw_summary,
        )

    @app.post("/users/{user_id}/search", response_model=SearchResponse)
    async def search_cross_session(user_id: str, body: CrossSessionSearchRequest):
        """Semantic search across all sessions for a user."""
        result = await manager.search_cross_session(user_id, body.query, top_k=body.top_k)
        return SearchResponse(
            records=[_record_out(r) for r in result.records],
            scores=result.scores,
        )

    @app.delete("/users/{user_id}", response_model=DeleteResponse)
    async def delete_user(user_id: str):
        """Delete all memories and profile for a user (GDPR)."""
        count = await manager.delete_user(user_id)
        return DeleteResponse(deleted=count)

    return app
