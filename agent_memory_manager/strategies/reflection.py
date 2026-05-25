from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from agent_memory_manager.models import MemoryRecord, MemoryType, Message
from agent_memory_manager.utils.json_utils import extract_json
from agent_memory_manager.utils.prompts import REFLECTION_PROMPT, MEMORY_CONTEXT_ITEM_TEMPLATE
from agent_memory_manager.utils.scoring import compute_retrieval_score
from agent_memory_manager.utils.token_counter import count_tokens

from .base import MemoryStrategy, ProcessResult

if TYPE_CHECKING:
    from agent_memory_manager.backends.base import MemoryBackend
    from agent_memory_manager.embedders.base import Embedder
    from agent_memory_manager.llm.base import LLMClient

logger = logging.getLogger(__name__)


class ReflectionStrategy(MemoryStrategy):
    """Synthesizes higher-order insights from episodic memories.

    Inspired by Generative Agents (Park et al., ACM UIST 2023).

    How it works:
    - After each ``process()`` call, importance scores of newly-added
      memories are accumulated per session.
    - When the accumulator exceeds ``reflection_threshold``, the strategy
      fetches the most recent ``recent_memory_limit`` memories, calls the
      LLM to synthesize insights, and stores them as
      ``MemoryType.REFLECTION`` records.
    - The accumulator is then reset to 0 for that session.

    Reflection records are included in context retrieval with elevated
    importance so they surface readily in future conversations.
    """

    def __init__(
        self,
        reflection_threshold: float = 150.0,
        max_insights: int = 5,
        recent_memory_limit: int = 20,
        delegate: MemoryStrategy | None = None,
    ) -> None:
        """
        Args:
            reflection_threshold: Sum of importance scores that triggers reflection.
            max_insights: Maximum number of insights to synthesize per reflection.
            recent_memory_limit: How many recent memories to consider for synthesis.
            delegate: Optional inner strategy that runs before reflection.
                      Typically AtomicFactsStrategy or SummarizeStrategy.
        """
        self.reflection_threshold = reflection_threshold
        self.max_insights = max_insights
        self.recent_memory_limit = recent_memory_limit
        self.delegate = delegate
        # Per-session importance accumulator (in-memory; resets on process restart)
        self._accumulators: dict[str, float] = {}

    async def process(
        self,
        messages: list[Message],
        session_id: str,
        backend: MemoryBackend,
        embedder: Embedder,
        llm: LLMClient,
    ) -> ProcessResult:
        result = ProcessResult()

        # 1. Run the inner delegate strategy first (if any)
        if self.delegate:
            inner = await self.delegate.process(messages, session_id, backend, embedder, llm)
            result.added.extend(inner.added)
            result.updated.extend(inner.updated)
            result.deleted.extend(inner.deleted)
            result.compressed = inner.compressed
            newly_added = inner.added
        else:
            newly_added = []

        # 2. Accumulate importance from newly added memories
        added_importance = sum(r.importance_score for r in newly_added)
        self._accumulators[session_id] = (
            self._accumulators.get(session_id, 0.0) + added_importance
        )

        # 3. Check if threshold is crossed
        if self._accumulators[session_id] < self.reflection_threshold:
            return result

        # 4. Fetch recent memories for synthesis
        recent = await backend.list_by_session(
            session_id, limit=self.recent_memory_limit
        )
        # Exclude existing reflection records from the synthesis input
        recent = [r for r in recent if r.memory_type != MemoryType.REFLECTION]
        if not recent:
            return result

        # 5. Synthesize insights via LLM
        insights = await self._synthesize(recent, llm)
        if not insights:
            self._accumulators[session_id] = 0.0
            return result

        # 6. Persist reflection records
        for insight_data in insights:
            content = insight_data.get("insight", "")
            importance = float(insight_data.get("importance", 7.0))
            evidence_indices: list[int] = insight_data.get("evidence_indices", [])

            if not content:
                continue

            source_ids = [
                recent[i].id for i in evidence_indices if 0 <= i < len(recent)
            ]
            record = MemoryRecord(
                session_id=session_id,
                memory_type=MemoryType.REFLECTION,
                content=f"[Reflection] {content}",
                source_message_ids=source_ids,
                importance_score=min(importance, 10.0),
            )
            record.embedding = await embedder.embed(content)
            await backend.save(record)
            result.added.append(record)

        # 7. Reset accumulator
        self._accumulators[session_id] = 0.0
        result.reflected = True

        logger.info(
            "Reflection triggered for session=%s: %d insights synthesized",
            session_id,
            len(result.added),
        )
        return result

    async def build_context(
        self,
        query: str,
        session_id: str,
        backend: MemoryBackend,
        embedder: Embedder,
        token_budget: int,
    ) -> str:
        # Delegate context building to inner strategy if present
        if self.delegate:
            return await self.delegate.build_context(
                query, session_id, backend, embedder, token_budget
            )

        query_embedding = await embedder.embed(query)
        candidates = await backend.search_by_vector(
            query_embedding, top_k=50, filters={"session_id": session_id}
        )

        scored = [
            (r, compute_retrieval_score(r, query_embedding))
            for r in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        lines: list[str] = []
        used = 0
        for record, _ in scored:
            line = MEMORY_CONTEXT_ITEM_TEMPLATE.format(content=record.content)
            t = count_tokens(line)
            if used + t > token_budget:
                break
            lines.append(line)
            used += t

        return "\n".join(lines)

    def get_accumulator(self, session_id: str) -> float:
        """Return current importance accumulator for a session."""
        return self._accumulators.get(session_id, 0.0)

    def reset_accumulator(self, session_id: str) -> None:
        """Manually reset the accumulator (e.g. on session delete)."""
        self._accumulators.pop(session_id, None)

    # ────────── private ──────────

    async def _synthesize(
        self,
        memories: list[MemoryRecord],
        llm: LLMClient,
    ) -> list[dict]:
        memories_text = "\n".join(
            f"[{i}] {r.content}" for i, r in enumerate(memories)
        )
        prompt = REFLECTION_PROMPT.format(
            max_insights=self.max_insights,
            recent_memories=memories_text,
        )
        try:
            response = await llm.generate(prompt, max_tokens=512, temperature=0.3)
            result = extract_json(response)
            return result if isinstance(result, list) else []
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("Reflection synthesis failed: %s", exc)
            return []
