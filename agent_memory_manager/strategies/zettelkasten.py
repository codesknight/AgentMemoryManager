from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from agent_memory_manager.models import MemoryRecord, MemoryType, Message
from agent_memory_manager.utils.json_utils import extract_json
from agent_memory_manager.utils.prompts import (
    ZETTELKASTEN_NOTE_PROMPT,
    IMPORTANCE_SCORING_PROMPT,
    MEMORY_CONTEXT_ITEM_TEMPLATE,
)
from agent_memory_manager.utils.scoring import cosine_similarity, compute_retrieval_score
from agent_memory_manager.utils.token_counter import count_tokens

from .base import MemoryStrategy, ProcessResult

if TYPE_CHECKING:
    from agent_memory_manager.backends.base import MemoryBackend
    from agent_memory_manager.embedders.base import Embedder
    from agent_memory_manager.llm.base import LLMClient

logger = logging.getLogger(__name__)


class ZettelkastenStrategy(MemoryStrategy):
    """Dynamic note-linking memory strategy inspired by A-MEM (arXiv:2502.12110, NeurIPS 2025).

    Each conversation turn becomes a structured note (content + keywords).
    Notes are automatically linked to semantically similar historical notes,
    forming a growing associative network that mirrors the Zettelkasten method.

    Key properties:
    - Each note is self-contained and atomic
    - Bidirectional links connect related notes
    - Context retrieval follows links for richer recall
    - The network evolves dynamically as new notes arrive
    """

    def __init__(
        self,
        link_threshold: float = 0.75,
        max_links_per_note: int = 5,
        link_hops: int = 1,
    ) -> None:
        """
        Args:
            link_threshold: Minimum cosine similarity to create a link between notes.
            max_links_per_note: Maximum number of outbound links per note.
            link_hops: How many link-hops to follow during context retrieval.
        """
        self.link_threshold = link_threshold
        self.max_links_per_note = max_links_per_note
        self.link_hops = link_hops

    async def process(
        self,
        messages: list[Message],
        session_id: str,
        backend: MemoryBackend,
        embedder: Embedder,
        llm: LLMClient,
    ) -> ProcessResult:
        result = ProcessResult()
        if not messages:
            return result

        # 1. Generate a structured note from the conversation turn
        note_data = await self._create_note(messages, llm)
        if not note_data:
            return result

        content: str = note_data.get("content", "")
        keywords: list[str] = note_data.get("keywords", [])

        if not content.strip():
            return result

        # 2. Score importance
        importance = await self._score_importance(content, llm)

        # 3. Embed the note content
        embedding = await embedder.embed(content)

        # 4. Find similar existing notes to link to
        candidates = await backend.search_by_vector(
            embedding, top_k=20, filters={"session_id": session_id}
        )
        links = [
            r.id
            for r in candidates
            if cosine_similarity(r.embedding or [], embedding) >= self.link_threshold
        ][: self.max_links_per_note]

        # 5. Create and save the new note
        record = MemoryRecord(
            session_id=session_id,
            memory_type=MemoryType.EPISODIC,
            content=content,
            source_message_ids=[m.id for m in messages],
            embedding=embedding,
            importance_score=importance,
            keywords=keywords,
            links=links,
        )
        await backend.save(record)
        result.added.append(record)

        # 6. Update existing linked notes to add a backlink (bidirectional)
        for linked_id in links:
            linked = await backend.get(linked_id)
            if linked and record.id not in linked.links:
                updated_links = linked.links + [record.id]
                await backend.update(linked_id, {"links": updated_links})

        logger.debug(
            "Zettelkasten note created session=%s links=%d",
            session_id,
            len(links),
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
        query_embedding = await embedder.embed(query)

        # 1. Direct semantic matches
        top_notes = await backend.search_by_vector(
            query_embedding, top_k=10, filters={"session_id": session_id}
        )

        # 2. Follow links up to `link_hops` hops
        seen_ids = {n.id for n in top_notes}
        frontier = list(top_notes)

        for _ in range(self.link_hops):
            next_frontier: list[MemoryRecord] = []
            for note in frontier:
                for lid in note.links:
                    if lid not in seen_ids:
                        linked = await backend.get(lid)
                        if linked:
                            next_frontier.append(linked)
                            seen_ids.add(lid)
            frontier = next_frontier

        all_notes = top_notes + frontier

        # 3. Re-rank by composite score
        scored = [
            (r, compute_retrieval_score(r, query_embedding))
            for r in all_notes
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        # 4. Build context within token budget
        lines: list[str] = []
        used = 0
        for record, _ in scored:
            kw = f" [{', '.join(record.keywords)}]" if record.keywords else ""
            line = MEMORY_CONTEXT_ITEM_TEMPLATE.format(content=f"{record.content}{kw}")
            t = count_tokens(line)
            if used + t > token_budget:
                break
            lines.append(line)
            used += t

        return "\n".join(lines)

    # ────────── private ──────────

    async def _create_note(
        self,
        messages: list[Message],
        llm: LLMClient,
    ) -> dict:
        conversation = "\n".join(
            f"{m.role.value.upper()}: {m.content}" for m in messages
        )
        prompt = ZETTELKASTEN_NOTE_PROMPT.format(conversation=conversation)
        try:
            response = await llm.generate(prompt, max_tokens=256, temperature=0.1)
            return extract_json(response)
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("Note creation failed, falling back to raw content: %s", exc)
            # Graceful fallback: store the raw message content as-is
            return {
                "content": " | ".join(m.content[:100] for m in messages),
                "keywords": [],
                "context": "",
            }

    async def _score_importance(self, content: str, llm: LLMClient) -> float:
        prompt = IMPORTANCE_SCORING_PROMPT.format(memory_content=content)
        try:
            response = await llm.generate(prompt, max_tokens=8, temperature=0.0)
            return max(1.0, min(10.0, float(response.strip())))
        except Exception:
            return 5.0
