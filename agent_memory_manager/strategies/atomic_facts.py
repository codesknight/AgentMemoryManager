from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from agent_memory_manager.models import MemoryRecord, MemoryType, Message
from agent_memory_manager.utils.token_counter import count_tokens
from agent_memory_manager.utils.prompts import (
    ATOMIC_FACTS_EXTRACTION_PROMPT,
    IMPORTANCE_SCORING_PROMPT,
    DEDUP_CHECK_PROMPT,
    MEMORY_CONTEXT_ITEM_TEMPLATE,
)
from agent_memory_manager.utils.scoring import compute_retrieval_score

from .base import MemoryStrategy, ProcessResult

if TYPE_CHECKING:
    from agent_memory_manager.backends.base import MemoryBackend
    from agent_memory_manager.embedders.base import Embedder
    from agent_memory_manager.llm.base import LLMClient

logger = logging.getLogger(__name__)


class AtomicFactsStrategy(MemoryStrategy):
    """Two-phase extraction pipeline inspired by Mem0 (arXiv:2504.19413).

    Phase 1 — Extract: LLM identifies atomic facts worth remembering from
    the new conversation turns.

    Phase 2 — Update: Each new fact is compared against existing memories.
    The LLM decides whether to ADD, UPDATE, DELETE, or SKIP to prevent
    duplicate or contradictory memories from accumulating.
    """

    def __init__(self, min_importance: float = 3.0) -> None:
        self.min_importance = min_importance

    async def process(
        self,
        messages: list[Message],
        session_id: str,
        backend: MemoryBackend,
        embedder: Embedder,
        llm: LLMClient,
    ) -> ProcessResult:
        result = ProcessResult()

        conversation = "\n".join(
            f"{m.role.value.upper()}: {m.content}" for m in messages
        )

        raw_facts = await self._extract_facts(conversation, llm)
        if not raw_facts:
            return result

        existing = await backend.list_by_session(session_id, limit=200)

        for fact_data in raw_facts:
            fact_text: str = fact_data.get("fact", "")
            importance: float = float(fact_data.get("importance", 5.0))

            if not fact_text or importance < self.min_importance:
                continue

            action, target_id = await self._decide_action(
                fact_text, existing, llm
            )

            if action == "skip":
                continue

            embedding = await embedder.embed(fact_text)

            if action == "add":
                record = MemoryRecord(
                    session_id=session_id,
                    memory_type=MemoryType.EPISODIC,
                    content=fact_text,
                    source_message_ids=[m.id for m in messages],
                    embedding=embedding,
                    importance_score=importance,
                )
                await backend.save(record)
                result.added.append(record)
                existing.append(record)

            elif action == "update" and target_id:
                await backend.update(
                    target_id,
                    {"content": fact_text, "embedding": embedding, "importance": importance},
                )
                updated = await backend.get(target_id)
                if updated:
                    result.updated.append(updated)

            elif action == "delete" and target_id:
                await backend.delete(target_id)
                result.deleted.append(target_id)
                existing = [r for r in existing if r.id != target_id]

        return result

    async def build_context(
        self,
        query: str,
        session_id: str,
        backend: MemoryBackend,
        embedder: Embedder,
        token_budget: int,
    ) -> str:
        query_embedding = await embedder.embed(query)
        candidates = await backend.search_by_vector(
            query_embedding,
            top_k=50,
            filters={"session_id": session_id},
        )

        # Re-rank with composite score (recency × importance × relevance)
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

    # ────────── private helpers ──────────

    async def _extract_facts(
        self, conversation: str, llm: LLMClient
    ) -> list[dict]:
        prompt = ATOMIC_FACTS_EXTRACTION_PROMPT.format(conversation=conversation)
        try:
            response = await llm.generate(prompt, max_tokens=512, temperature=0.0)
            return json.loads(response)
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("Failed to parse extracted facts: %s", exc)
            return []

    async def _decide_action(
        self,
        new_fact: str,
        existing: list[MemoryRecord],
        llm: LLMClient,
    ) -> tuple[str, str | None]:
        if not existing:
            return "add", None

        existing_text = "\n".join(
            f"[{i}] (id={r.id}) {r.content}" for i, r in enumerate(existing[:30])
        )
        prompt = DEDUP_CHECK_PROMPT.format(
            new_fact=new_fact,
            existing_memories=existing_text,
        )
        try:
            response = await llm.generate(prompt, max_tokens=64, temperature=0.0)
            data = json.loads(response)
            return data.get("action", "add"), data.get("target_id")
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("Dedup check failed, defaulting to add: %s", exc)
            return "add", None
