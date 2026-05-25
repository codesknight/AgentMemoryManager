"""Unit tests for scoring utilities."""
import pytest
from datetime import datetime, timedelta, timezone

from agent_memory_manager.utils.scoring import (
    cosine_similarity,
    compute_recency,
    compute_retrieval_score,
)
from agent_memory_manager.models import MemoryRecord


def test_cosine_similarity_identical():
    v = [1.0, 0.0, 0.0]
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_empty():
    assert cosine_similarity([], [1.0]) == 0.0


def test_recency_fresh():
    now = datetime.now(timezone.utc)
    score = compute_recency(now)
    assert score > 0.99  # Just accessed — very high


def test_recency_old():
    old = datetime.now(timezone.utc) - timedelta(days=7)
    score = compute_recency(old, half_life_hours=24.0)
    assert score < 0.1  # A week old — low


def test_retrieval_score_ordering():
    """Higher-importance memory should score higher when recency/relevance are equal."""
    base_time = datetime.now(timezone.utc)
    r_high = MemoryRecord(session_id="s", content="x", importance_score=9.0)
    r_high.accessed_at = base_time
    r_high.embedding = [1.0, 0.0]

    r_low = MemoryRecord(session_id="s", content="y", importance_score=2.0)
    r_low.accessed_at = base_time
    r_low.embedding = [1.0, 0.0]

    query_emb = [1.0, 0.0]
    s_high = compute_retrieval_score(r_high, query_emb)
    s_low = compute_retrieval_score(r_low, query_emb)
    assert s_high > s_low
