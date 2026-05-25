"""Unit tests for StreamingCompressStrategy."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agent_memory_manager.strategies.streaming import StreamingCompressStrategy
from agent_memory_manager.strategies.base import ProcessResult
from agent_memory_manager.models import Message, MemoryRecord, MemoryType
from agent_memory_manager.models.message import Role


def _make_record(content: str, session_id: str = "s1") -> MemoryRecord:
    return MemoryRecord(
        session_id=session_id,
        content=content,
        memory_type=MemoryType.EPISODIC,
    )


def _make_msg(content: str, role: Role = Role.USER) -> Message:
    return Message(role=role, content=content)


def _make_backend(records: list[MemoryRecord] | None = None) -> MagicMock:
    backend = AsyncMock()
    backend.list_by_session = AsyncMock(return_value=records or [])
    backend.save = AsyncMock()
    backend.delete = AsyncMock(return_value=True)
    return backend


def _make_embedder() -> MagicMock:
    embedder = AsyncMock()
    embedder.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
    return embedder


def _make_llm(summary: str = "Summary of old messages.") -> MagicMock:
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=summary)
    return llm


class TestStreamingCompressStrategy:
    """Tests for StreamingCompressStrategy."""

    @pytest.mark.asyncio
    async def test_process_below_threshold_no_compression(self):
        strategy = StreamingCompressStrategy(compress_threshold=1000)
        # Few records well below threshold
        existing = [_make_record("short msg " * 2) for _ in range(3)]
        backend = _make_backend(existing)
        embedder = _make_embedder()
        llm = _make_llm()

        msgs = [_make_msg("hello world")]
        result = await strategy.process(msgs, "s1", backend, embedder, llm)

        assert isinstance(result, ProcessResult)
        assert len(result.added) == 1
        assert result.deleted == []
        assert result.compressed is False
        llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_stores_incoming_messages(self):
        strategy = StreamingCompressStrategy(compress_threshold=9999)
        backend = _make_backend([])
        embedder = _make_embedder()
        llm = _make_llm()

        msgs = [_make_msg("msg1"), _make_msg("msg2", Role.ASSISTANT)]
        result = await strategy.process(msgs, "s1", backend, embedder, llm)

        assert backend.save.call_count == 2
        assert len(result.added) == 2

    @pytest.mark.asyncio
    async def test_process_compresses_when_over_threshold(self):
        strategy = StreamingCompressStrategy(
            compress_threshold=10,   # very low threshold
            preserve_recent=2,
            max_summary_tokens=50,
        )
        # 8 existing records, enough to exceed threshold
        existing = [_make_record(f"long content message number {i} " * 5) for i in range(8)]
        backend = _make_backend(existing)
        embedder = _make_embedder()
        llm = _make_llm("Compressed summary of old messages.")

        msgs = [_make_msg("new message")]
        result = await strategy.process(msgs, "s1", backend, embedder, llm)

        assert result.compressed is True
        # Old records deleted (all except preserve_recent=2)
        assert len(result.deleted) == len(existing) - 2
        # Summary record added
        summary_records = [r for r in result.added if "Compressed summary" in r.content]
        assert len(summary_records) == 1
        assert summary_records[0].importance_score == 7.0

    @pytest.mark.asyncio
    async def test_process_llm_failure_skips_compression(self):
        strategy = StreamingCompressStrategy(compress_threshold=5, preserve_recent=1)
        existing = [_make_record("x " * 20) for _ in range(5)]
        backend = _make_backend(existing)
        embedder = _make_embedder()
        llm = AsyncMock()
        llm.generate = AsyncMock(side_effect=RuntimeError("LLM down"))

        msgs = [_make_msg("new")]
        result = await strategy.process(msgs, "s1", backend, embedder, llm)

        # No deletion when LLM fails
        assert result.compressed is False
        assert result.deleted == []
        backend.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_embed_failure_is_graceful(self):
        strategy = StreamingCompressStrategy(compress_threshold=9999)
        backend = _make_backend([])
        embedder = AsyncMock()
        embedder.embed = AsyncMock(side_effect=Exception("embed error"))
        llm = _make_llm()

        msgs = [_make_msg("test")]
        # Should not raise
        result = await strategy.process(msgs, "s1", backend, embedder, llm)
        assert len(result.added) == 1

    @pytest.mark.asyncio
    async def test_process_preserves_recent_messages(self):
        strategy = StreamingCompressStrategy(
            compress_threshold=10,
            preserve_recent=3,
        )
        existing = [_make_record(f"msg {i} " * 10) for i in range(6)]
        backend = _make_backend(existing)
        embedder = _make_embedder()
        llm = _make_llm("Summary.")

        msgs = [_make_msg("newest")]
        result = await strategy.process(msgs, "s1", backend, embedder, llm)

        # Should delete 6 - 3 = 3 old records
        assert len(result.deleted) == 3

    @pytest.mark.asyncio
    async def test_build_context_empty_returns_empty_string(self):
        strategy = StreamingCompressStrategy()
        backend = _make_backend([])
        embedder = _make_embedder()

        ctx = await strategy.build_context("query", "s1", backend, embedder, 500)
        assert ctx == ""

    @pytest.mark.asyncio
    async def test_build_context_returns_ranked_content(self):
        strategy = StreamingCompressStrategy()
        records = [
            _make_record("relevant content about topic A"),
            _make_record("irrelevant content about bananas"),
        ]
        for r in records:
            r.embedding = [0.9, 0.1]
        backend = _make_backend(records)
        embedder = _make_embedder()

        ctx = await strategy.build_context("topic A", "s1", backend, embedder, 1000)
        assert "relevant content" in ctx

    @pytest.mark.asyncio
    async def test_build_context_embed_failure_falls_back_to_recency(self):
        strategy = StreamingCompressStrategy()
        records = [_make_record("msg A"), _make_record("msg B")]
        backend = _make_backend(records)
        embedder = AsyncMock()
        embedder.embed = AsyncMock(side_effect=Exception("fail"))

        ctx = await strategy.build_context("anything", "s1", backend, embedder, 1000)
        assert "msg" in ctx

    @pytest.mark.asyncio
    async def test_too_few_records_skips_compression(self):
        strategy = StreamingCompressStrategy(
            compress_threshold=10,
            preserve_recent=5,
        )
        # Only 3 records but threshold exceeded — preserve_recent=5 keeps all
        existing = [_make_record("x " * 10) for _ in range(3)]
        backend = _make_backend(existing)
        embedder = _make_embedder()
        llm = _make_llm("Summary.")

        msgs = [_make_msg("new")]
        result = await strategy.process(msgs, "s1", backend, embedder, llm)

        assert result.compressed is False
        llm.generate.assert_not_called()
