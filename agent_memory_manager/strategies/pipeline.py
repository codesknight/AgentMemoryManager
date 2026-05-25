from __future__ import annotations

from typing import TYPE_CHECKING

from .base import MemoryStrategy, ProcessResult

if TYPE_CHECKING:
    from agent_memory_manager.backends.base import MemoryBackend
    from agent_memory_manager.embedders.base import Embedder
    from agent_memory_manager.llm.base import LLMClient
    from agent_memory_manager.models import Message


class StrategyPipeline(MemoryStrategy):
    """Chains multiple strategies sequentially.

    Each strategy's ProcessResult is merged into a single accumulated result.
    Context building delegates to the last strategy in the pipeline.
    """

    def __init__(self, strategies: list[MemoryStrategy]) -> None:
        if not strategies:
            raise ValueError("StrategyPipeline requires at least one strategy")
        self.strategies = strategies

    async def process(
        self,
        messages: list["Message"],
        session_id: str,
        backend: "MemoryBackend",
        embedder: "Embedder",
        llm: "LLMClient",
    ) -> ProcessResult:
        merged = ProcessResult()
        for strategy in self.strategies:
            result = await strategy.process(messages, session_id, backend, embedder, llm)
            merged.added.extend(result.added)
            merged.updated.extend(result.updated)
            merged.deleted.extend(result.deleted)
            merged.compressed = merged.compressed or result.compressed
            merged.reflected = merged.reflected or result.reflected
        return merged

    async def build_context(
        self,
        query: str,
        session_id: str,
        backend: "MemoryBackend",
        embedder: "Embedder",
        token_budget: int,
    ) -> str:
        # Delegate to the last strategy (usually the richest one)
        return await self.strategies[-1].build_context(
            query, session_id, backend, embedder, token_budget
        )
