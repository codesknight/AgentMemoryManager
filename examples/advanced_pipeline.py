"""Advanced Pipeline Example — Zettelkasten + Reflection + Graph Memory (v1.5).

Demonstrates a production-style setup combining:
  1. ZettelkastenStrategy — structured note creation with auto-linking
  2. ReflectionStrategy   — periodic synthesis of high-level insights
  3. Graph Memory (v1.5)  — automatic entity/relation extraction via MemoryManager

Run:
    python examples/advanced_pipeline.py
"""

import asyncio
import json

from agent_memory_manager import MemoryManager, Message, Role
from agent_memory_manager.backends import InMemoryBackend
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
        # Graph entity/relation extraction — detected by prompt header line
        if "Extract named entities and their relationships" in prompt:
            return json.dumps({
                "entities": [
                    {"name": "Sam", "type": "person", "attributes": {"role": "ML engineer"}},
                    {"name": "DataCo", "type": "organization", "attributes": {}},
                    {"name": "RAG Pipeline", "type": "project", "attributes": {"stack": "Python"}},
                ],
                "relations": [
                    {"subject": "Sam", "predicate": "works_at", "object": "DataCo", "confidence": 0.95},
                    {"subject": "Sam", "predicate": "builds", "object": "RAG Pipeline", "confidence": 0.9},
                ],
            })
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
    print("=== Advanced Pipeline: Zettelkasten + Reflection + Graph Memory (v1.5) ===\n")

    session_id = "adv-demo-001"

    # ── 1. Build the strategy pipeline ──────────────────────────────────────
    pipeline = StrategyPipeline([
        ZettelkastenStrategy(link_threshold=0.6, max_links_per_note=3),
        ReflectionStrategy(reflection_threshold=5.0, max_insights=3),
    ])

    manager = MemoryManager(
        backend=InMemoryBackend(),
        strategy=pipeline,
        llm=MockLLM(),
        embedder=MockEmbedder(),
        enable_graph=True,   # v1.5: automatic graph extraction on every add()
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
        print(
            f"  Turn {i+1}: +{len(result.added)} added  "
            f"reflected={result.reflected}  "
            f"entities={result.entities_extracted}  "
            f"relations={result.relations_extracted}"
        )

    # ── 3. Graph API (v1.5) ──────────────────────────────────────────────────
    print("\n── Graph Memory (v1.5) ──")

    # Query neighbourhood of an entity
    graph_result = await manager.query_graph("Sam", session_id=session_id, hops=1)
    print(f"  Sam's neighbours ({len(graph_result.neighbours)}):")
    for n in graph_result.neighbours:
        target = n["entity"].name if n["entity"] else "?"
        print(f"    → {n['relation']} → {target} (confidence={n['confidence']:.2f})")

    # Fetch a single entity
    sam_entity = await manager.get_entity("Sam", session_id=session_id)
    if sam_entity:
        print(f"\n  Entity 'Sam': type={sam_entity.entity_type}, attrs={sam_entity.attributes}")

    # List all persons
    persons = await manager.list_entities(session_id, entity_type="person")
    print(f"\n  Persons in graph: {[e.name for e in persons]}")

    # ── 4. Memory-enhanced prompt ────────────────────────────────────────────
    print("\nBuilding memory-enhanced prompt...")
    prompt = await manager.build_prompt(
        "What ML projects is this user working on?",
        session_id,
        token_budget=800,
    )
    print(prompt)

    # ── 5. Stats (include graph counts) ─────────────────────────────────────
    stats = await manager.get_stats(session_id)
    print(f"\nMemory stats:")
    print(f"  Total records   : {stats.total_memories}")
    print(f"  Episodic        : {stats.episodic_count}")
    print(f"  Reflections     : {stats.reflection_count}")
    print(f"  Graph entities  : {stats.graph_entity_count}")
    print(f"  Graph relations : {stats.graph_relation_count}")
    print(f"  Est. tokens     : {stats.estimated_tokens}")

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
