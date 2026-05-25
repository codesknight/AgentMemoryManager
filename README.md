# AgentMemoryManager

A pluggable memory management component for LLM-based agents, solving context degradation in long conversations and multi-turn tasks.

## The Problem

LLMs suffer from **context degradation** as conversations grow longer:
- Relevant information buried in the middle gets ignored (accuracy drops >30%)
- Token costs grow linearly with conversation history
- Cross-session memory is completely lost

## The Solution

AgentMemoryManager provides a **four-layer cognitive memory architecture** inspired by how human memory works:

```
Working Memory    → Active context window management (compression, sliding window)
Episodic Memory   → Persistent atomic facts extracted from conversations
Semantic Memory   → Entity-relationship knowledge graph (v1.5)
Procedural Memory → Reusable task templates and tool-use patterns
```

## Quick Start

```python
from agent_memory_manager import MemoryManager, MemoryConfig
from agent_memory_manager.backends import SQLiteBackend
from agent_memory_manager.llm import AnthropicClient
from agent_memory_manager.embedders import LocalEmbedder

manager = MemoryManager.from_config(
    MemoryConfig(
        backend="sqlite",
        llm_provider="anthropic",
        llm_model="claude-sonnet-4-6",
    )
)

# Add conversation turns
await manager.add(messages=[...], session_id="user-123")

# Retrieve memory-enhanced prompt
enhanced_prompt = await manager.build_prompt(
    base_prompt="What was the project I mentioned earlier?",
    session_id="user-123",
)
```

## Installation

```bash
pip install agent-memory-manager

# With optional backends / integrations
pip install agent-memory-manager[chroma]      # Chroma vector DB
pip install agent-memory-manager[qdrant]      # Qdrant vector DB
pip install agent-memory-manager[ollama]      # Local Ollama LLM + embedder
pip install agent-memory-manager[langchain]   # LangChain integration
pip install agent-memory-manager[all]         # Everything
```

## Quick Start — Local LLM with Ollama

Run a fully local pipeline without any API keys.

**Prerequisites**

```bash
# Install Ollama: https://ollama.com
ollama pull qwen3:0.6b          # or any chat model
ollama pull nomic-embed-text    # dedicated embedding model (recommended)

pip install agent-memory-manager[ollama]
```

**Code**

```python
import asyncio
from agent_memory_manager import MemoryManager, Message, Role
from agent_memory_manager.backends import InMemoryBackend
from agent_memory_manager.llm.openai import OpenAIClient
from agent_memory_manager.embedders.ollama_embedder import OllamaEmbedder
from agent_memory_manager.strategies import AtomicFactsStrategy

async def main():
    manager = MemoryManager(
        backend=InMemoryBackend(),
        strategy=AtomicFactsStrategy(),
        llm=OpenAIClient(
            model="qwen3:0.6b",
            base_url="http://localhost:11434/v1",
            keep_alive=0,       # release GPU after each call (recommended for shared VRAM)
        ),
        embedder=OllamaEmbedder(model="nomic-embed-text"),
    )
    await manager.initialize()

    msgs = [
        Message(role=Role.USER, content="Hi, I'm Sam, senior ML engineer at DataCo."),
        Message(role=Role.ASSISTANT, content="Nice to meet you Sam!"),
    ]
    result = await manager.add(messages=msgs, session_id="demo")
    print(f"Stored {len(result.added)} memories")

    prompt = await manager.build_prompt("What does this user do?", "demo")
    print(prompt)

asyncio.run(main())
```

> **Windows + system proxy note**: `OpenAIClient` and `OllamaEmbedder` default to
> `trust_env=False`, bypassing Clash / V2Ray / WinINet proxy so that local Ollama
> calls are not intercepted. For **external APIs** (OpenAI, Doubao, etc.) that need
> the system proxy, pass `trust_env=True`:
> ```python
> OpenAIClient(model="...", api_key="...", base_url="...", trust_env=True)
> ```

## Graph Memory (v1.5)

Automatically extract entities and relations from every conversation turn and query the resulting knowledge graph.

```python
manager = MemoryManager(
    backend=SQLiteBackend("memory.db"),
    strategy=AtomicFactsStrategy(),
    llm=OpenAIClient(model="claude-sonnet-4-6"),
    embedder=LocalEmbedder(),
    enable_graph=True,          # auto-extract on every add()
    graph_db_path="graph.db",   # persist graph to SQLite (optional)
)
await manager.initialize()

# Entities and relations are extracted automatically during add()
result = await manager.add(
    messages=[Message(role=Role.USER, content="I'm Sam, ML engineer at DataCo.")],
    session_id="user-123",
)
print(f"Extracted {result.entities_extracted} entities, {result.relations_extracted} relations")

# Query the knowledge graph
graph = await manager.query_graph("Sam", session_id="user-123", hops=1)
for n in graph.neighbours:
    print(f"  Sam --[{n['relation']}]--> {n['entity'].name}")

# Fetch a single entity with its attributes
entity = await manager.get_entity("Sam", session_id="user-123")
print(entity.attributes)   # {'role': 'ML engineer'}

# List entities by type
persons = await manager.list_entities("user-123", entity_type="person")

# Graph counts appear in get_stats()
stats = await manager.get_stats("user-123")
print(stats.graph_entity_count, stats.graph_relation_count)
```

## Architecture

```
┌─────────────────────────────────────────────┐
│              Your Application               │
│     LangChain / LlamaIndex / Custom Agent   │
└────────────────────┬────────────────────────┘
                     │ Python SDK
┌────────────────────▼────────────────────────┐
│           AgentMemoryManager Core            │
│  MemoryManager → StrategyEngine → Layers    │
│  GraphExtractor → SemanticMemory (v1.5)     │
└────────────────────┬────────────────────────┘
                     │
┌────────────────────▼────────────────────────┐
│           Storage Backends                  │
│  InMemory │ SQLite │ Chroma │ Qdrant │ PG   │
└────────────────────────────────────────────┘
```

## Key Features

- **Pluggable strategies**: Sliding window, summarization, atomic fact extraction, reflection, Zettelkasten
- **Multiple backends**: In-memory, SQLite, Chroma, Qdrant, PostgreSQL+pgvector
- **Graph Memory** (v1.5): Automatic entity/relation extraction, multi-hop graph queries, SQLite persistence
- **Multi-provider LLM support**: Anthropic Claude, OpenAI, Ollama (local), any OpenAI-compatible API
- **Framework integrations**: LangChain, LlamaIndex
- **Production-ready**: GDPR-compliant deletion, 118 unit tests, 86% coverage

## Benchmarks

| Approach | Accuracy | P95 Latency | Tokens/Session |
|----------|----------|-------------|----------------|
| Full context (baseline) | 72.9% | 9.87s | ~26,000 |
| **AgentMemoryManager** | ≥ 65% | < 2s | < 4,000 |

*Evaluated on LOCOMO benchmark (ACL 2024)*

## Research Foundation

Built on top of frontier research (2023–2025):
- **Mem0** (arXiv:2504.19413) — atomic fact extraction pipeline
- **Generative Agents** (Park et al., UIST 2023) — reflection mechanism
- **A-MEM** (arXiv:2502.12110, NeurIPS 2025) — Zettelkasten dynamic linking
- **Zep/Graphiti** (arXiv:2501.13956) — temporal knowledge graph design
- **StreamingLLM** (ICLR 2024) — attention sink management
- **LLMLingua** (EMNLP 2023) — token-level compression

## Roadmap

- [x] v0.1 — Technical research & design documents
- [x] v1.0 — Core strategies (SlidingWindow / Summarize / AtomicFacts / Reflection / Zettelkasten), all backends, LangChain/LlamaIndex, CI
- [x] v1.0.1 — Robust JSON parsing, prompt optimization, OllamaEmbedder, `trust_env` for external APIs
- [x] v1.5 — Graph Memory: `GraphExtractor`, `GraphStore` (SQLite), `query_graph` / `get_entity` / `list_entities` API
- [ ] v2.0 — Cross-session user memory, REST API server, pgvector backend, streaming compression

## License

MIT
