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
User Memory       → Cross-session user profiles synthesized by LLM (v2.0)
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
pip install agent-memory-manager[pgvector]    # PostgreSQL + pgvector
pip install agent-memory-manager[server]      # REST API (FastAPI + Uvicorn)
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
```

## Cross-Session User Memory (v2.0)

Build a persistent user profile that survives across multiple sessions. The LLM synthesizes facts and preferences from all past conversations.

```python
manager = MemoryManager(
    backend=SQLiteBackend("memory.db"),
    strategy=AtomicFactsStrategy(),
    llm=OpenAIClient(model="claude-sonnet-4-6"),
    embedder=LocalEmbedder(),
    user_profile_db_path="profiles.db",   # enable profile persistence
)
await manager.initialize()

# Add memories across multiple sessions under the same user_id
await manager.add(messages=[...], session_id="session-1", user_id="alice")
await manager.add(messages=[...], session_id="session-2", user_id="alice")

# Synthesize a user profile from all sessions (LLM-powered)
profile = await manager.build_user_profile("alice")
print(profile.facts)          # ["Alice is a backend engineer", ...]
print(profile.preferences)    # {"language": "Python", "style": "concise"}
print(profile.raw_summary)    # narrative summary

# Semantic search across all sessions for this user
result = await manager.search_cross_session("alice", query="database preferences")
for r in result.records:
    print(r.session_id, r.content)

# Delete all data for a user (GDPR)
deleted_count = await manager.delete_user("alice")
```

## REST API Server (v2.0)

Deploy AgentMemoryManager as a standalone HTTP service using FastAPI.

```bash
# Docker (recommended)
docker-compose up

# Or run directly
pip install agent-memory-manager[server]
AMM_BACKEND=sqlite AMM_DB_PATH=/data/memory.db \
AMM_LLM_MODEL=gpt-4o-mini AMM_LLM_API_KEY=$OPENAI_API_KEY \
uvicorn agent_memory_manager.server.entrypoint:app --host 0.0.0.0 --port 8000
```

**Available endpoints**

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/sessions/{id}/memories` | Add messages to a session |
| `GET` | `/sessions/{id}/memories` | List all memories for a session |
| `POST` | `/sessions/{id}/search` | Semantic search in a session |
| `POST` | `/sessions/{id}/prompt` | Build a memory-enhanced prompt |
| `DELETE` | `/sessions/{id}` | Delete all memories for a session |
| `POST` | `/users/{id}/profile` | Build / refresh user profile |
| `GET` | `/users/{id}/profile` | Get cached user profile |
| `POST` | `/users/{id}/search` | Cross-session semantic search |
| `DELETE` | `/users/{id}` | Delete all user data |
| `GET` | `/sessions/{id}/graph/{entity}` | Query knowledge graph |
| `GET` | `/stats` | Server statistics |

**Embed in your own FastAPI app**

```python
from fastapi import FastAPI
from agent_memory_manager.server import create_app

app: FastAPI = create_app(manager)   # manager is your MemoryManager instance
```

## Streaming Compression (v2.0)

`StreamingCompressStrategy` compresses context incrementally — every new message is stored immediately, and when the total token count exceeds a threshold the oldest messages are summarized and replaced. This keeps `build_prompt()` latency low regardless of conversation length.

```python
from agent_memory_manager.strategies import StreamingCompressStrategy

manager = MemoryManager(
    backend=SQLiteBackend("memory.db"),
    strategy=StreamingCompressStrategy(
        compress_threshold=800,   # token count that triggers compression
        preserve_recent=4,        # always keep the N most recent messages verbatim
        max_summary_tokens=200,   # max tokens for the LLM summary
    ),
    llm=OpenAIClient(model="gpt-4o-mini"),
    embedder=LocalEmbedder(),
)
```

## PostgreSQL + pgvector Backend (v2.0)

Production-grade vector storage with native ANN search via the pgvector `<=>` cosine operator and IVFFlat indexing.

```bash
# Requires PostgreSQL ≥ 14 with pgvector enabled
# CREATE EXTENSION IF NOT EXISTS vector;

pip install agent-memory-manager[pgvector]
```

```python
from agent_memory_manager.backends import PgVectorBackend

backend = PgVectorBackend(
    dsn="postgresql://user:password@localhost:5432/mydb",
    vector_dim=384,   # must match your embedder's output dimension
    pool_size=10,
)
manager = MemoryManager(backend=backend, ...)
await manager.initialize()   # creates table + IVFFlat index automatically
```

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                  Your Application                     │
│        LangChain / LlamaIndex / Custom Agent          │
└───────────────────────┬──────────────────────────────┘
                        │ Python SDK  or  HTTP (v2.0)
┌───────────────────────▼──────────────────────────────┐
│          REST API Server (v2.0)                       │
│          FastAPI · create_app(manager)                │
└───────────────────────┬──────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────────┐
│           AgentMemoryManager Core                     │
│  MemoryManager → StrategyEngine → Memory Layers      │
│  GraphExtractor → SemanticMemory (v1.5)              │
│  UserProfileStore → Cross-session profiles (v2.0)    │
└───────────────────────┬──────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────────┐
│              Storage Backends                         │
│  InMemory │ SQLite │ Chroma │ Qdrant │ pgvector (v2.0)│
└──────────────────────────────────────────────────────┘
```

## Key Features

- **Pluggable strategies**: Sliding window, summarization, atomic facts, reflection, Zettelkasten, streaming compression (v2.0)
- **Multiple backends**: In-memory, SQLite, Chroma, Qdrant, PostgreSQL+pgvector (v2.0)
- **Graph Memory** (v1.5): Automatic entity/relation extraction, multi-hop graph queries, SQLite persistence
- **Cross-session user memory** (v2.0): LLM-synthesized user profiles, cross-session semantic search, GDPR deletion
- **REST API** (v2.0): FastAPI server with Docker support, embeddable `create_app()` factory
- **Multi-provider LLM support**: Anthropic Claude, OpenAI, Ollama (local), any OpenAI-compatible API
- **Framework integrations**: LangChain, LlamaIndex
- **Production-ready**: GDPR-compliant deletion, 174 unit tests, 86% coverage

## Benchmarks

Token compression benchmark on a synthetic multi-turn conversation dataset
(see `tests/benchmarks/compression_benchmark.py`). Numbers reflect
GPT-4o-mini-quality extraction; small local models yield lower ratios
due to weaker JSON output reliability.

| Conversation length | Original tokens | Compressed tokens | Compression ratio |
|---------------------|-----------------|-------------------|-------------------|
| 12 turns  (short)   | 131             | 69                | **47%**           |
| 40 turns  (medium)  | 484             | 162               | **67%**           |
| 80 turns  (long)    | 1,044           | 239               | **77%**           |
| 100+ turns (production multi-session) | — | — | **85%+** (estimated) |

> **Key insight**: compression ratio scales with conversation length and topic
> repetition. AtomicFacts' dedup phase removes repeated mentions of the same
> facts; longer sessions with returning users benefit most.
>
> The 85%+ figure cited in the project summary refers to extended production
> sessions (100+ turns) where the same user facts are mentioned many times
> across turns. Short conversations (< 20 turns) yield lower but still
> meaningful compression (~47–55%).

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
- [x] v2.0 — Cross-session user memory, REST API server, pgvector backend, streaming compression
- [ ] v2.1 — Multi-tenant auth, async batch ingestion, OpenTelemetry tracing

## License

MIT
