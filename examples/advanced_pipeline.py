"""Advanced Pipeline Example — Zettelkasten + Reflection + Semantic Memory.

Demonstrates a production-style setup combining:
  1. ZettelkastenStrategy — structured note creation with auto-linking
  2. ReflectionStrategy   — periodic synthesis of high-level insights
  3. SemanticMemory        — entity-relation knowledge graph

Run:
    python examples/advanced_pipeline.py
"""

import asyncio
import json
from unittest.mock import AsyncMock

from agent_memory_manager import MemoryManager, Message, Role
from agent_memory_manager.backends import InMemoryBackend
from agent_memory_manager.memory import SemanticMemory
from agent_memory_manager.models.entity import Entity, Relation
from agent_memory_manager.strategies import (
    ReflectionStrategy,
    StrategyPipeline,
    ZettelkastenStrategy,
)


# ─── Mock embedder: hash-based deterministic vectors ────────────────────────
class MockEmbedder:
    dimensions = 64

    async def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dimensions
        for i, ch in enumerate(text.lower()):
            vec[i % self.dimensions] += ord(ch) / 10_000.0
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]

    async def embed_batch(self, texts):
        return [await self.embed(t) for t in texts]


# ─── Mock LLM: returns structured responses based on prompt type ─────────────
class MockLLM:
    _call_count = 0

    async def generate(self, prompt: str, **kwargs) -> str:
        self._call_count += 1
        # Zettelkasten note creation
        if "Zettelkasten" in prompt or "structured note" in prompt.lower():
            return json.dumps({
                "content": "User is a senior ML engineer building a RAG pipeline at DataCo.",
                "keywords": ["ml", "rag", "dataco"],
                "context": "User disclosed professional context and current project.",
            })
        # Importance scoring
        if "Rate the importance" in prompt:
            return "8"
        # Reflection synthesis
        if "synthesize" in prompt.lower() or "insights" in prompt.lower():
            return json.dumps([{
                "insight": "User consistently works on ML infrastructure projects.",
                "evidence_indices": [0, 1],
                "importance": 9,
            }])
        return "[]"

    async def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


async def main():
    print("=== Advanced Pipeline: Zettelkasten + Reflection + SemanticMemory ===\n")

    session_id = "adv-demo-001"

    # ── 1. Build the strategy pipeline ──────────────────────────────────────
    pipeline = StrategyPipeline([
        ZettelkastenStrategy(link_threshold=0.6, max_links_per_note=3),
        ReflectionStrategy(reflection_threshold=5.0, max_insights=3),  # low threshold for demo
    ])

    manager = MemoryManager(
        backend=InMemoryBackend(),
        strategy=pipeline,
        llm=MockLLM(),
        embedder=MockEmbedder(),
    )
    await manager.initialize()

    # ── 2. Feed a multi-turn conversation ────────────────────────────────────
    conversations = [
        [
            Message(role=Role.USER, content="Hi, I'm Sam, senior ML engineer at DataCo."),
            Message(role=Role.ASSISTANT, content="Welcome Sam! What are you working on?"),
        ],
        [
            Message(role=Role.USER, content="We're building a RAG pipeline for internal docs."),
            Message(role=Role.ASSISTANT, content="RAG is powerful for knowledge retrieval."),
        ],
        [
            Message(role=Role.USER, content="We use Python and prefer open-source tools."),
            Message(role=Role.ASSISTANT, content="Great choices for flexibility."),
        ],
    ]

    print("Feeding conversation turns...")
    for i, turn in enumerate(conversations):
        result = await manager.add(messages=turn, session_id=session_id)
        print(f"  Turn {i+1}: +{len(result.added)} added  reflected={result.reflected}")

    # ── 3. Retrieve memory-enhanced context ─────────────────────────────────
    print("\nBuilding memory-enhanced prompt...")
    prompt = await manager.build_prompt(
        "What ML projects is this user working on?",
        session_id,
        token_budget=800,
    )
    print(prompt)

    # ── 4. Print stats ───────────────────────────────────────────────────────
    stats = await manager.get_stats(session_id)
    print(f"\nMemory stats:")
    print(f"  Total records : {stats.total_memories}")
    print(f"  Episodic      : {stats.episodic_count}")
    print(f"  Reflections   : {stats.reflection_count}")
    print(f"  Est. tokens   : {stats.estimated_tokens}")

    # ── 5. Demonstrate SemanticMemory (standalone) ───────────────────────────
    print("\n── SemanticMemory (Knowledge Graph) ──")
    sm = SemanticMemory(session_id)

    sam = Entity(session_id=session_id, name="Sam", entity_type="person",
                 attributes={"role": "ML engineer"})
    dataco = Entity(session_id=session_id, name="DataCo", entity_type="organization")
    rag_project = Entity(session_id=session_id, name="RAG Pipeline", entity_type="project")

    sm.add_entity(sam)
    sm.add_entity(dataco)
    sm.add_entity(rag_project)

    sm.add_relation(Relation(
        session_id=session_id, subject_id="Sam",
        predicate="works_at", object_id="DataCo"
    ))
    sm.add_relation(Relation(
        session_id=session_id, subject_id="Sam",
        predicate="builds", object_id="RAG Pipeline"
    ))

    neighbours = sm.get_neighbours("Sam", hops=1)
    print(f"  Sam's direct connections:")
    for n in neighbours:
        entity_name = n["entity"].name if n["entity"] else "?"
        print(f"    → {n['relation']} → {entity_name} (confidence={n['confidence']:.2f})")

    two_hop = sm.get_neighbours("Sam", hops=2)
    print(f"  2-hop reachable entities: {[n['entity'].name for n in two_hop if n['entity']]}")

    print(f"\n  Graph: {sm.entity_count} entities, {sm.relation_count} relations")
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
