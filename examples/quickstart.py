"""
AgentMemoryManager — Quickstart Example

Demonstrates the full lifecycle using:
  - InMemoryBackend (no DB setup needed)
  - SlidingWindowStrategy (no LLM calls needed)
  - A mock embedder (no API key needed)

Run:
    python examples/quickstart.py
"""

import asyncio
from unittest.mock import AsyncMock

from agent_memory_manager import MemoryManager, MemoryConfig, Message, Role
from agent_memory_manager.backends import InMemoryBackend
from agent_memory_manager.strategies import SlidingWindowStrategy


class MockEmbedder:
    """Zero-dependency embedder for demo purposes."""
    dimensions = 4

    async def embed(self, text: str) -> list[float]:
        # Hash-based mock embedding (not useful for real retrieval)
        h = hash(text) % 1000
        return [h / 1000, (h * 7 % 1000) / 1000, (h * 13 % 1000) / 1000, 0.5]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


class MockLLM:
    async def generate(self, prompt: str, **kwargs) -> str:
        return "[]"

    async def count_tokens(self, text: str) -> int:
        return len(text) // 4


async def main():
    print("=== AgentMemoryManager Quickstart ===\n")

    manager = MemoryManager(
        backend=InMemoryBackend(),
        strategy=SlidingWindowStrategy(window_size=10),
        llm=MockLLM(),
        embedder=MockEmbedder(),
    )
    await manager.initialize()

    session_id = "demo-user-001"

    # ── Round 1: introduce yourself ──
    print("Round 1: User introduction")
    await manager.add(
        messages=[
            Message(role=Role.USER, content="Hi! I'm Alex, a data scientist at TechCorp."),
            Message(role=Role.ASSISTANT, content="Nice to meet you, Alex!"),
        ],
        session_id=session_id,
    )

    # ── Round 2: share project context ──
    print("Round 2: Project context")
    await manager.add(
        messages=[
            Message(role=Role.USER, content="I'm building a RAG pipeline for internal docs."),
            Message(role=Role.ASSISTANT, content="That sounds interesting!"),
        ],
        session_id=session_id,
    )

    # ── Round 3: ask something that requires memory ──
    print("\nRound 3: Memory-enhanced prompt")
    enhanced = await manager.build_prompt(
        base_prompt="What tech stack would you recommend for my project?",
        session_id=session_id,
        token_budget=500,
    )
    print(enhanced)

    # ── Stats ──
    stats = await manager.get_stats(session_id)
    print(f"\nMemory stats: {stats.total_memories} records, "
          f"~{stats.estimated_tokens} tokens")

    # ── Cleanup (GDPR) ──
    deleted = await manager.delete_session(session_id)
    print(f"\nDeleted {deleted} memories for session.")


if __name__ == "__main__":
    asyncio.run(main())
