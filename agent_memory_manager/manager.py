from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from .backends.base import MemoryBackend
from .config import MemoryConfig
from .embedders.base import Embedder
from .llm.base import LLMClient
from .models import Message, MemoryRecord, MemoryStats, MemoryType
from .models.results import AddResult, CompressionResult, ContextResult, SearchResult
from .strategies.base import MemoryStrategy
from .utils.prompts import MEMORY_INJECTION_TEMPLATE
from .utils.scoring import compute_retrieval_score
from .utils.token_counter import count_tokens

logger = logging.getLogger(__name__)


class MemoryManager:
    """Main entry point for AgentMemoryManager.

    Manages the full memory lifecycle: ingestion, storage, retrieval,
    compression, and context injection. All I/O is async.

    Typical usage::

        manager = MemoryManager(backend, strategy, llm, embedder, config)
        await manager.initialize()

        await manager.add(messages=[...], session_id="user-123")
        prompt = await manager.build_prompt("What was my project?", "user-123")
    """

    def __init__(
        self,
        backend: MemoryBackend,
        strategy: MemoryStrategy,
        llm: LLMClient,
        embedder: Embedder,
        config: Optional[MemoryConfig] = None,
    ) -> None:
        self._backend = backend
        self._strategy = strategy
        self._llm = llm
        self._embedder = embedder
        self._config = config or MemoryConfig()

        if self._config.enable_logging:
            logging.basicConfig(level=self._config.log_level)

    # ────────── Lifecycle ──────────

    async def initialize(self) -> None:
        """Initialize storage backend (create tables, connect, etc.)."""
        await self._backend.initialize()
        logger.debug("MemoryManager initialized with backend=%s", type(self._backend).__name__)

    async def close(self) -> None:
        """Release all resources."""
        await self._backend.close()

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
        """Process new conversation messages and update memory."""
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

        logger.info(
            "add session=%s added=%d updated=%d deleted=%d",
            session_id,
            len(result.added),
            len(result.updated),
            len(result.deleted),
        )
        return AddResult(
            added=result.added,
            updated=result.updated,
            deleted=result.deleted,
            compressed=result.compressed,
            reflected=result.reflected,
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
        # Note: multi-type filter is done in Python for simplicity
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

        # Update accessed_at for retrieved records
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

    # ────────── Manage ──────────

    async def compress(self, session_id: str) -> CompressionResult:
        """Manually trigger memory compression for a session."""
        before = await self._backend.count(session_id)
        records = await self._backend.list_by_session(session_id, limit=10_000)
        before_tokens = sum(r.token_estimate() for r in records)

        # Re-run strategy's process with empty messages to trigger compression
        from .strategies.summarize import SummarizeStrategy
        compressor = SummarizeStrategy(
            summarize_threshold=0,  # Force compression
            preserve_recent=self._config.preserve_recent_turns,
        )
        result = await compressor.process([], session_id, self._backend, self._embedder, self._llm)

        after = await self._backend.count(session_id)
        records_after = await self._backend.list_by_session(session_id, limit=10_000)
        after_tokens = sum(r.token_estimate() for r in records_after)

        return CompressionResult(
            original_token_count=before_tokens,
            compressed_token_count=after_tokens,
            memories_deleted=len(result.deleted),
            summaries_created=len(result.added),
        )

    async def delete_session(self, session_id: str) -> int:
        """Delete all memories for a session (GDPR right-to-be-forgotten)."""
        count = await self._backend.delete_by_session(session_id)
        logger.info("Deleted %d memories for session=%s", count, session_id)
        return count

    async def get_stats(self, session_id: str) -> MemoryStats:
        """Return memory statistics for a session."""
        records = await self._backend.list_by_session(session_id, limit=10_000)
        if not records:
            return MemoryStats(session_id=session_id)

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
        )


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
