from __future__ import annotations

from typing import TYPE_CHECKING

from agent_memory_manager.models import MemoryRecord, MemoryType, Message
from agent_memory_manager.utils.token_counter import count_tokens
from agent_memory_manager.utils.prompts import SUMMARIZE_PROMPT, MEMORY_CONTEXT_ITEM_TEMPLATE

from .base import MemoryStrategy, ProcessResult

if TYPE_CHECKING:
    from agent_memory_manager.backends.base import MemoryBackend
    from agent_memory_manager.embedders.base import Embedder
    from agent_memory_manager.llm.base import LLMClient


class SummarizeStrategy(MemoryStrategy):
    """Compresses old conversation turns into a rolling summary.

    When the buffered history exceeds summarize_threshold tokens, the oldest
    turns (beyond preserve_recent) are summarized and replaced with a single
    EPISODIC memory record. The summary is re-embedded for vector retrieval.
    """

    def __init__(
        self,
        summarize_threshold: int = 2000,
        preserve_recent: int = 5,
    ) -> None:
        self.summarize_threshold = summarize_threshold
        self.preserve_recent = preserve_recent

    async def process(
        self,
        messages: list[Message],
        session_id: str,
        backend: MemoryBackend,
        embedder: Embedder,
        llm: LLMClient,
    ) -> ProcessResult:
        result = ProcessResult()

        for msg in messages:
            record = MemoryRecord(
                session_id=session_id,
                memory_type=MemoryType.EPISODIC,
                content=f"{msg.role.value}: {msg.content}",
                source_message_ids=[msg.id],
                importance_score=5.0,
            )
            record.embedding = await embedder.embed(msg.content)
            await backend.save(record)
            result.added.append(record)

        # Check whether we need to compress
        all_records = await backend.list_by_session(session_id, limit=10_000)
        total_tokens = sum(r.token_estimate() for r in all_records)

        if total_tokens > self.summarize_threshold:
            compress_result = await self._compress(
                all_records, session_id, backend, embedder, llm
            )
            result.deleted.extend(compress_result.deleted)
            result.added.extend(compress_result.added)
            result.compressed = True

        return result

    async def _compress(
        self,
        records: list[MemoryRecord],
        session_id: str,
        backend: MemoryBackend,
        embedder: Embedder,
        llm: LLMClient,
    ) -> ProcessResult:
        result = ProcessResult()
        to_compress = records[: -self.preserve_recent] if len(records) > self.preserve_recent else records
        if not to_compress:
            return result

        conversation_text = "\n".join(r.content for r in to_compress)
        summary = await llm.generate(
            SUMMARIZE_PROMPT.format(conversation=conversation_text),
            max_tokens=256,
        )

        summary_record = MemoryRecord(
            session_id=session_id,
            memory_type=MemoryType.EPISODIC,
            content=f"[Summary] {summary}",
            source_message_ids=[r.id for r in to_compress],
            importance_score=7.0,
        )
        summary_record.embedding = await embedder.embed(summary)
        await backend.save(summary_record)
        result.added.append(summary_record)

        for r in to_compress:
            await backend.delete(r.id)
            result.deleted.append(r.id)

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
        records = await backend.search_by_vector(
            query_embedding,
            top_k=20,
            filters={"session_id": session_id},
        )

        lines: list[str] = []
        used = 0
        for record in records:
            line = MEMORY_CONTEXT_ITEM_TEMPLATE.format(content=record.content)
            t = count_tokens(line)
            if used + t > token_budget:
                break
            lines.append(line)
            used += t

        return "\n".join(lines)
