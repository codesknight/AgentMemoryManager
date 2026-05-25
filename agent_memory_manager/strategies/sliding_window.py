from __future__ import annotations

from typing import TYPE_CHECKING

from agent_memory_manager.models import MemoryRecord, MemoryType, Message
from agent_memory_manager.utils.token_counter import count_tokens
from agent_memory_manager.utils.prompts import MEMORY_CONTEXT_ITEM_TEMPLATE

from .base import MemoryStrategy, ProcessResult

if TYPE_CHECKING:
    from agent_memory_manager.backends.base import MemoryBackend
    from agent_memory_manager.embedders.base import Embedder
    from agent_memory_manager.llm.base import LLMClient


class SlidingWindowStrategy(MemoryStrategy):
    """Keeps the most recent N conversation turns in the context window.

    Zero LLM calls — the simplest and fastest strategy.
    Best for short tasks where losing early history is acceptable.
    """

    def __init__(self, window_size: int = 20) -> None:
        self.window_size = window_size

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
            await backend.save(record)
            result.added.append(record)
        return result

    async def build_context(
        self,
        query: str,
        session_id: str,
        backend: MemoryBackend,
        embedder: Embedder,
        token_budget: int,
    ) -> str:
        all_records = await backend.list_by_session(
            session_id, limit=self.window_size * 2
        )
        # Take the most recent window_size records
        recent = all_records[-self.window_size :]

        lines: list[str] = []
        used_tokens = 0
        for record in reversed(recent):  # Newest first, then trim to budget
            line = MEMORY_CONTEXT_ITEM_TEMPLATE.format(content=record.content)
            tokens = count_tokens(line)
            if used_tokens + tokens > token_budget:
                break
            lines.insert(0, line)
            used_tokens += tokens

        return "\n".join(lines)
