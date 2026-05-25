from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class MemoryConfig(BaseModel):
    # ── LLM ────────────────────────────────────────────────────────────
    llm_provider: Literal["anthropic", "openai", "litellm"] = "anthropic"
    llm_model: str = "claude-sonnet-4-6"
    llm_temperature: float = 0.0
    llm_api_key: Optional[str] = None

    # ── Embedder ────────────────────────────────────────────────────────
    embedder: Literal["openai", "local"] = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedder_api_key: Optional[str] = None

    # ── Storage backend ─────────────────────────────────────────────────
    backend: Literal["in_memory", "sqlite", "chroma", "qdrant"] = "in_memory"
    backend_url: Optional[str] = None  # file path for sqlite, URL for others

    # ── Memory layers ────────────────────────────────────────────────────
    enable_working_memory: bool = True
    enable_episodic_memory: bool = True
    enable_semantic_memory: bool = False
    enable_procedural_memory: bool = False

    # ── Working memory ────────────────────────────────────────────────────
    working_memory_token_budget: int = 4000
    working_memory_strategy: Literal["sliding_window", "summarize"] = "sliding_window"
    sliding_window_size: int = 20

    # ── Episodic memory ───────────────────────────────────────────────────
    episodic_strategy: Literal["atomic_facts", "sliding_window", "summarize"] = "atomic_facts"
    summarize_threshold: int = 3000
    preserve_recent_turns: int = 5
    reflection_threshold: float = 150.0

    # ── Retrieval ──────────────────────────────────────────────────────────
    retrieval_top_k: int = 10
    retrieval_token_budget: int = 2000
    retrieval_weights: tuple[float, float, float] = Field(default=(1.0, 1.0, 1.0))

    # ── Observability ──────────────────────────────────────────────────────
    enable_logging: bool = True
    log_level: str = "INFO"
