from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_memory_manager.models import MemoryRecord


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_recency(
    last_accessed: datetime,
    half_life_hours: float = 24.0,
) -> float:
    """Exponential decay: score = 0.99^(hours_elapsed).
    At half_life_hours the score is approximately 0.5."""
    now = datetime.now(timezone.utc)
    hours_elapsed = (now - last_accessed).total_seconds() / 3600
    decay_rate = 0.5 ** (1 / half_life_hours)
    return decay_rate ** hours_elapsed


def compute_retrieval_score(
    record: "MemoryRecord",
    query_embedding: list[float],
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """Composite retrieval score from Generative Agents (Park et al., 2023).

    score = α·recency + β·importance + γ·relevance
    """
    alpha, beta, gamma = weights

    recency = compute_recency(record.accessed_at)
    importance = record.importance_score / 10.0
    relevance = (
        cosine_similarity(record.embedding, query_embedding)
        if record.embedding and query_embedding
        else 0.0
    )

    return alpha * recency + beta * importance + gamma * relevance
