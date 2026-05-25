"""StreamingCompressStrategy — real-time context compression (v2.0-C).

Instead of compressing after the fact, this strategy compresses incrementally:
each new message is immediately evaluated and either stored as-is (if under budget)
or summarized with nearby messages before storage, keeping token counts low
regardless of conversation length.

This reduces first-token latency for build_prompt() because the stored context
is always pre-compressed and ready to inject.
"""
from __future__ import annotations

import logging
from typing import Optional

from .base import MemoryStrategy, ProcessResult
from agent_memory_manager.models import Message, MemoryRecord, MemoryType
from agent_memory_manager.utils.token_counter import count_tokens
from agent_memory_manager.utils.prompts import SUMMARIZE_PROMPT

logger = logging.getLogger(__name__)


class StreamingCompressStrategy(MemoryStrategy):
    """Incrementally compress conversation context as messages arrive.

    Every call to ``process()`` accumulates messages into a rolling buffer.
    When the buffer exceeds ``compress_threshold`` tokens, the oldest messages
    are summarized with the LLM and replaced by a single summary record.
    The most recent ``preserve_recent`` messages are always kept verbatim.

    Args:
        compress_threshold: Token count that triggers compression (default 800).
        preserve_recent:    Number of recent messages to keep uncompressed (default 4).
        max_summary_tokens: Maximum tokens for the LLM summary output (default 200).
    """

    def __init__(
        self,
        compress_threshold: int = 800,
        preserve_recent: int = 4,
        max_summary_tokens: int = 200,
    ) -> None:
        self._threshold = compress_threshold
        self._preserve = preserve_recent
        self._max_summary_tokens = max_summary_tokens

    async def process(
        self,
        messages: list[Message],
        session_id: str,
        backend,
        embedder,
        llm,
    ) -> StrategyResult:
        added: list[MemoryRecord] = []
        deleted: list[str] = []

        # Embed and store incoming messages
        for msg in messages:
            record = MemoryRecord(
                session_id=session_id,
                content=f"{msg.role.value.upper()}: {msg.content}",
                memory_type=MemoryType.EPISODIC,
            )
            try:
                record.embedding = await embedder.embed(record.content)
            except Exception:
                pass
            await backend.save(record)
            added.append(record)

        # Check if compression is needed
        all_records = await backend.list_by_session(session_id, limit=10_000)
        total_tokens = sum(count_tokens(r.content) for r in all_records)

        if total_tokens <= self._threshold or len(all_records) <= self._preserve:
            return ProcessResult(added=added)

        # Split: keep recent, compress old
        to_compress = all_records[: -self._preserve]
        conversation = "\n".join(r.content for r in to_compress)

        summary_text: Optional[str] = None
        try:
            summary_text = await llm.generate(
                SUMMARIZE_PROMPT.format(conversation=conversation),
                max_tokens=self._max_summary_tokens,
                temperature=0.0,
            )
        except Exception as exc:
            logger.warning("StreamingCompressStrategy: LLM failed: %s", exc)

        if summary_text:
            # Delete old records
            for r in to_compress:
                await backend.delete(r.id)
                deleted.append(r.id)

            # Store summary as a single record
            summary_record = MemoryRecord(
                session_id=session_id,
                content=summary_text.strip(),
                memory_type=MemoryType.EPISODIC,
                importance_score=7.0,
            )
            try:
                summary_record.embedding = await embedder.embed(summary_record.content)
            except Exception:
                pass
            await backend.save(summary_record)
            added.append(summary_record)

            logger.info(
                "StreamingCompress session=%s compressed=%d→1 tokens=%d→%d",
                session_id, len(to_compress),
                total_tokens, count_tokens(summary_text),
            )

        return ProcessResult(added=added, deleted=deleted, compressed=bool(summary_text))

    async def build_context(
        self,
        query: str,
        session_id: str,
        backend,
        embedder,
        token_budget: int,
    ) -> str:
        from agent_memory_manager.utils.scoring import cosine_similarity
        from agent_memory_manager.utils.token_counter import truncate_to_budget
        from agent_memory_manager.utils.prompts import MEMORY_CONTEXT_ITEM_TEMPLATE

        try:
            query_emb = await embedder.embed(query)
        except Exception:
            query_emb = []

        records = await backend.list_by_session(session_id, limit=10_000)
        if not records:
            return ""

        if query_emb:
            scored = sorted(
                records,
                key=lambda r: cosine_similarity(r.embedding or [], query_emb),
                reverse=True,
            )
        else:
            scored = sorted(records, key=lambda r: r.created_at, reverse=True)

        items = [MEMORY_CONTEXT_ITEM_TEMPLATE.format(content=r.content) for r in scored]
        return "\n".join(truncate_to_budget(items, token_budget))
