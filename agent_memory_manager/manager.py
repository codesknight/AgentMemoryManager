from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from .backends.base import MemoryBackend
from .config import MemoryConfig
from .embedders.base import Embedder
from .llm.base import LLMClient
from .memory.graph_extractor import GraphExtractor
from .memory.graph_store import GraphStore
from .memory.semantic_memory import SemanticMemory
from .memory.user_profile_store import UserProfileStore
from .models import Message, MemoryRecord, MemoryStats, MemoryType
from .models.entity import Entity
from .models.results import (
    AddResult, CompressionResult, ContextResult,
    GraphQueryResult, SearchResult,
)
from .models.user_profile import UserProfile
from .strategies.base import MemoryStrategy
from .utils.prompts import MEMORY_INJECTION_TEMPLATE, USER_PROFILE_SYNTHESIS_PROMPT
from .utils.scoring import compute_retrieval_score
from .utils.token_counter import count_tokens

logger = logging.getLogger(__name__)


class MemoryManager:
    """Main entry point for AgentMemoryManager.

    Manages the full memory lifecycle: ingestion, storage, retrieval,
    compression, context injection, and knowledge-graph extraction.
    All I/O is async.

    Typical usage::

        manager = MemoryManager(backend, strategy, llm, embedder, config)
        await manager.initialize()

        await manager.add(messages=[...], session_id="user-123")
        prompt = await manager.build_prompt("What was my project?", "user-123")

        # v1.5: knowledge graph
        result = await manager.query_graph("Sam", session_id="user-123")
        entity  = await manager.get_entity("Sam", session_id="user-123")
    """

    def __init__(
        self,
        backend: MemoryBackend,
        strategy: MemoryStrategy,
        llm: LLMClient,
        embedder: Embedder,
        config: Optional[MemoryConfig] = None,
        enable_graph: bool = True,
        graph_db_path: Optional[str] = None,
        user_profile_db_path: Optional[str] = None,
    ) -> None:
        self._backend = backend
        self._strategy = strategy
        self._llm = llm
        self._embedder = embedder
        self._config = config or MemoryConfig()
        self._enable_graph = enable_graph

        # Per-session knowledge graphs (in-process cache)
        self._graphs: dict[str, SemanticMemory] = {}
        self._extractor = GraphExtractor()
        self._graph_store: Optional[GraphStore] = (
            GraphStore(graph_db_path) if graph_db_path else None
        )

        # Cross-session user profiles
        self._profile_store: Optional[UserProfileStore] = (
            UserProfileStore(user_profile_db_path) if user_profile_db_path else None
        )
        self._user_profiles: dict[str, UserProfile] = {}

        if self._config.enable_logging:
            logging.basicConfig(level=self._config.log_level)

    # ────────── Lifecycle ──────────

    async def initialize(self) -> None:
        """Initialize storage backend, graph store, and user profile store."""
        await self._backend.initialize()
        if self._graph_store:
            await self._graph_store.initialize()
        if self._profile_store:
            await self._profile_store.initialize()
        logger.debug("MemoryManager initialized with backend=%s", type(self._backend).__name__)

    async def close(self) -> None:
        """Release all resources."""
        await self._backend.close()
        if self._graph_store:
            await self._graph_store.close()
        if self._profile_store:
            await self._profile_store.close()

    # ────────── Factory ──────────

    @classmethod
    def from_config(cls, config: MemoryConfig) -> "MemoryManager":
        """Convenience factory that wires up components from a MemoryConfig."""
        backend = _build_backend(config)
        strategy = _build_strategy(config)
        llm = _build_llm(config)
        embedder = _build_embedder(config)
        return cls(backend, strategy, llm, embedder, config)

    # ────────── Write ──────────

    async def add(
        self,
        messages: list[Message],
        session_id: str,
        user_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> AddResult:
        """Process new conversation messages and update memory.

        When ``enable_graph=True`` (default), also runs LLM entity extraction
        and populates the session knowledge graph.
        """
        if not messages:
            return AddResult()

        if user_id:
            for m in messages:
                if not m.metadata.get("user_id"):
                    m.metadata["user_id"] = user_id
        if metadata:
            for m in messages:
                m.metadata.update(metadata)

        result = await self._strategy.process(
            messages=messages,
            session_id=session_id,
            backend=self._backend,
            embedder=self._embedder,
            llm=self._llm,
        )

        # ── Knowledge-graph extraction ──
        entities_n = relations_n = 0
        if self._enable_graph:
            graph = await self._get_or_create_graph(session_id)
            entities_n, relations_n = await self._extractor.extract(
                messages, session_id, graph, self._llm
            )
            if self._graph_store and (entities_n or relations_n):
                await self._graph_store.save(graph)

        logger.info(
            "add session=%s added=%d updated=%d deleted=%d entities=%d relations=%d",
            session_id,
            len(result.added), len(result.updated), len(result.deleted),
            entities_n, relations_n,
        )
        return AddResult(
            added=result.added,
            updated=result.updated,
            deleted=result.deleted,
            compressed=result.compressed,
            reflected=result.reflected,
            entities_extracted=entities_n,
            relations_extracted=relations_n,
        )

    # ────────── Retrieve ──────────

    async def search(
        self,
        query: str,
        session_id: str,
        top_k: Optional[int] = None,
        memory_types: Optional[list[MemoryType]] = None,
    ) -> SearchResult:
        """Semantic search over stored memories."""
        k = top_k or self._config.retrieval_top_k
        query_embedding = await self._embedder.embed(query)

        filters: dict = {"session_id": session_id}
        candidates = await self._backend.search_by_vector(
            query_embedding, top_k=k * 3, filters=filters
        )

        if memory_types:
            candidates = [r for r in candidates if r.memory_type in memory_types]

        scored = [
            (r, compute_retrieval_score(r, query_embedding, self._config.retrieval_weights))
            for r in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:k]

        now = datetime.now(timezone.utc)
        for record, _ in top:
            await self._backend.update(record.id, {"accessed_at": now})

        return SearchResult(
            records=[r for r, _ in top],
            scores=[s for _, s in top],
        )

    async def build_context(
        self,
        query: str,
        session_id: str,
        token_budget: Optional[int] = None,
    ) -> ContextResult:
        """Build a formatted memory context string for prompt injection."""
        budget = token_budget or self._config.retrieval_token_budget
        context = await self._strategy.build_context(
            query=query,
            session_id=session_id,
            backend=self._backend,
            embedder=self._embedder,
            token_budget=budget,
        )
        return ContextResult(
            context=context,
            token_count=count_tokens(context),
        )

    async def build_prompt(
        self,
        base_prompt: str,
        session_id: str,
        token_budget: Optional[int] = None,
    ) -> str:
        """Return base_prompt with relevant memories injected."""
        ctx = await self.build_context(base_prompt, session_id, token_budget)
        if not ctx.context.strip():
            return base_prompt
        return MEMORY_INJECTION_TEMPLATE.format(
            memory_context=ctx.context,
            base_prompt=base_prompt,
        )

    # ────────── Graph ──────────

    async def query_graph(
        self,
        entity_name: str,
        session_id: str,
        hops: int = 1,
        current_only: bool = True,
    ) -> GraphQueryResult:
        """Query the knowledge graph for an entity's neighbourhood.

        Args:
            entity_name: Name of the entity to query (case-insensitive).
            session_id:  Session whose graph to search.
            hops:        How many relation-hops to traverse (default 1).
            current_only: If True, skip expired relations.

        Returns:
            GraphQueryResult with neighbours list and graph stats.
        """
        graph = self._graphs.get(session_id)
        if graph is None:
            return GraphQueryResult(entity_name=entity_name)

        neighbours = graph.get_neighbours(entity_name, hops=hops, current_only=current_only)
        return GraphQueryResult(
            entity_name=entity_name,
            neighbours=neighbours,
            total_entities=graph.entity_count,
            total_relations=graph.relation_count,
        )

    async def get_entity(
        self,
        entity_name: str,
        session_id: str,
    ) -> Optional[Entity]:
        """Retrieve a single entity from the session knowledge graph."""
        graph = self._graphs.get(session_id)
        if graph is None:
            return None
        return graph.get_entity(entity_name)

    async def list_entities(
        self,
        session_id: str,
        entity_type: Optional[str] = None,
    ) -> list[Entity]:
        """List all entities in the session knowledge graph, optionally filtered by type."""
        graph = self._graphs.get(session_id)
        if graph is None:
            return []
        return graph.search_entities(entity_type=entity_type)

    # ────────── User Memory (v2.0) ──────────

    async def build_user_profile(
        self,
        user_id: str,
        force_rebuild: bool = False,
    ) -> UserProfile:
        """Synthesize a UserProfile from all memories tagged with user_id.

        The profile is cached in memory and optionally persisted to SQLite.
        Pass ``force_rebuild=True`` to re-synthesize even if a cached version exists.

        Returns:
            UserProfile with deduplicated facts, preferences, and a raw summary.
        """
        if not force_rebuild:
            if user_id in self._user_profiles:
                return self._user_profiles[user_id]
            if self._profile_store:
                cached = await self._profile_store.load(user_id)
                if cached:
                    self._user_profiles[user_id] = cached
                    return cached

        records = await self._backend.list_by_user(user_id)
        session_ids = list({r.session_id for r in records})
        facts_text = "\n".join(f"- {r.content}" for r in records[:200])  # cap at 200

        profile = UserProfile(
            user_id=user_id,
            session_ids=session_ids,
            total_memories=len(records),
        )

        if records and facts_text:
            try:
                from .utils.json_utils import extract_json
                raw = await self._llm.generate(
                    USER_PROFILE_SYNTHESIS_PROMPT.format(facts=facts_text),
                    max_tokens=512,
                    temperature=0.0,
                )
                parsed = extract_json(raw)
                if isinstance(parsed, dict):
                    profile.facts = parsed.get("facts", [])
                    profile.preferences = parsed.get("preferences", {})
                    profile.raw_summary = parsed.get("raw_summary", "")
            except Exception as exc:
                logger.warning("UserProfile synthesis failed for %s: %s", user_id, exc)

        self._user_profiles[user_id] = profile
        if self._profile_store:
            await self._profile_store.save(profile)

        logger.info(
            "Built user profile: user=%s sessions=%d memories=%d facts=%d",
            user_id, len(session_ids), len(records), len(profile.facts),
        )
        return profile

    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Return a cached or persisted user profile without rebuilding."""
        if user_id in self._user_profiles:
            return self._user_profiles[user_id]
        if self._profile_store:
            return await self._profile_store.load(user_id)
        return None

    async def search_cross_session(
        self,
        user_id: str,
        query: str,
        top_k: Optional[int] = None,
    ) -> SearchResult:
        """Semantic search across all sessions for a given user_id.

        Unlike ``search()``, this is not scoped to a single session.
        """
        k = top_k or self._config.retrieval_top_k
        query_embedding = await self._embedder.embed(query)

        candidates = await self._backend.search_by_vector(
            query_embedding,
            top_k=k * 3,
            filters={"user_id": user_id},
        )

        scored = [
            (r, compute_retrieval_score(r, query_embedding, self._config.retrieval_weights))
            for r in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:k]

        now = datetime.now(timezone.utc)
        for record, _ in top:
            await self._backend.update(record.id, {"accessed_at": now})

        return SearchResult(
            records=[r for r, _ in top],
            scores=[s for _, s in top],
        )

    async def delete_user(self, user_id: str) -> int:
        """Delete all memories and profile for a user (GDPR). Returns count deleted."""
        count = await self._backend.delete_by_user(user_id)
        self._user_profiles.pop(user_id, None)
        if self._profile_store:
            await self._profile_store.delete(user_id)
        logger.info("Deleted %d memories for user=%s", count, user_id)
        return count

    # ────────── Manage ──────────

    async def compress(self, session_id: str) -> CompressionResult:
        """Manually trigger memory compression for a session."""
        records = await self._backend.list_by_session(session_id, limit=10_000)
        before_tokens = sum(r.token_estimate() for r in records)

        from .strategies.summarize import SummarizeStrategy
        compressor = SummarizeStrategy(
            summarize_threshold=0,
            preserve_recent=self._config.preserve_recent_turns,
        )
        result = await compressor.process([], session_id, self._backend, self._embedder, self._llm)

        records_after = await self._backend.list_by_session(session_id, limit=10_000)
        after_tokens = sum(r.token_estimate() for r in records_after)

        return CompressionResult(
            original_token_count=before_tokens,
            compressed_token_count=after_tokens,
            memories_deleted=len(result.deleted),
            summaries_created=len(result.added),
        )

    async def delete_session(self, session_id: str) -> int:
        """Delete all memories and graph data for a session (GDPR)."""
        count = await self._backend.delete_by_session(session_id)
        self._graphs.pop(session_id, None)
        if self._graph_store:
            await self._graph_store.delete(session_id)
        logger.info("Deleted %d memories for session=%s", count, session_id)
        return count

    async def get_stats(self, session_id: str) -> MemoryStats:
        """Return memory statistics for a session."""
        records = await self._backend.list_by_session(session_id, limit=10_000)
        graph = self._graphs.get(session_id)

        if not records:
            return MemoryStats(
                session_id=session_id,
                graph_entity_count=graph.entity_count if graph else 0,
                graph_relation_count=graph.relation_count if graph else 0,
            )

        now = datetime.now(timezone.utc)
        oldest_hours = max(
            (now - r.created_at).total_seconds() / 3600 for r in records
        )
        avg_importance = sum(r.importance_score for r in records) / len(records)

        counts = {t: 0 for t in MemoryType}
        for r in records:
            counts[r.memory_type] += 1

        return MemoryStats(
            session_id=session_id,
            total_memories=len(records),
            episodic_count=counts[MemoryType.EPISODIC],
            semantic_count=counts[MemoryType.SEMANTIC],
            reflection_count=counts[MemoryType.REFLECTION],
            procedural_count=counts[MemoryType.PROCEDURAL],
            estimated_tokens=sum(r.token_estimate() for r in records),
            oldest_memory_age_hours=oldest_hours,
            avg_importance_score=avg_importance,
            graph_entity_count=graph.entity_count if graph else 0,
            graph_relation_count=graph.relation_count if graph else 0,
        )

    # ────────── Internal ──────────

    async def _get_or_create_graph(self, session_id: str) -> SemanticMemory:
        if session_id not in self._graphs:
            # Try to restore from SQLite first
            if self._graph_store:
                loaded = await self._graph_store.load(session_id)
                if loaded:
                    self._graphs[session_id] = loaded
                    return loaded
            self._graphs[session_id] = SemanticMemory(session_id)
        return self._graphs[session_id]


# ────────── Factory helpers ──────────

def _build_backend(config: MemoryConfig) -> MemoryBackend:
    if config.backend == "in_memory":
        from .backends.in_memory import InMemoryBackend
        return InMemoryBackend()
    if config.backend == "sqlite":
        from .backends.sqlite import SQLiteBackend
        return SQLiteBackend(db_path=config.backend_url or "memory.db")
    raise ValueError(f"Unknown backend: {config.backend!r}")


def _build_strategy(config: MemoryConfig) -> MemoryStrategy:
    if config.episodic_strategy == "atomic_facts":
        from .strategies.atomic_facts import AtomicFactsStrategy
        return AtomicFactsStrategy()
    if config.episodic_strategy == "summarize":
        from .strategies.summarize import SummarizeStrategy
        return SummarizeStrategy(
            summarize_threshold=config.summarize_threshold,
            preserve_recent=config.preserve_recent_turns,
        )
    from .strategies.sliding_window import SlidingWindowStrategy
    return SlidingWindowStrategy(window_size=config.sliding_window_size)


def _build_llm(config: MemoryConfig) -> LLMClient:
    if config.llm_provider == "anthropic":
        from .llm.anthropic import AnthropicClient
        return AnthropicClient(model=config.llm_model, api_key=config.llm_api_key)
    if config.llm_provider == "openai":
        from .llm.openai import OpenAIClient
        return OpenAIClient(model=config.llm_model, api_key=config.llm_api_key)
    raise ValueError(f"Unknown LLM provider: {config.llm_provider!r}")


def _build_embedder(config: MemoryConfig) -> Embedder:
    if config.embedder == "openai":
        from .embedders.openai_embedder import OpenAIEmbedder
        return OpenAIEmbedder(model=config.embedding_model, api_key=config.embedder_api_key)
    if config.embedder == "local":
        from .embedders.local_embedder import LocalEmbedder
        return LocalEmbedder(model=config.embedding_model)
    raise ValueError(f"Unknown embedder: {config.embedder!r}")
