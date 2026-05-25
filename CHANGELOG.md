# Changelog

All notable changes to AgentMemoryManager are documented here.

## [2.0.0] — 2026-05-25

### Added

**A — Cross-Session User Memory**
- `UserProfile` dataclass — aggregates facts, preferences, session IDs, and raw summary per user
- `UserProfileStore` — async SQLite persistence (`user_profiles` table); `save`, `load`, `delete`, `list_users`
- `MemoryManager.build_user_profile(user_id, force_rebuild)` — LLM-synthesized profile from all sessions
- `MemoryManager.get_user_profile(user_id)` — retrieve cached profile
- `MemoryManager.search_cross_session(user_id, query, top_k)` — semantic search across all user sessions
- `MemoryManager.delete_user(user_id)` — purge all memories, graph, and profile for a user
- `user_profile_db_path` constructor param on `MemoryManager`
- `USER_PROFILE_SYNTHESIS_PROMPT` — structured prompt for profile synthesis
- `list_by_user` / `delete_by_user` on `InMemoryBackend`, `SQLiteBackend`, and `PgVectorBackend`
- `user_id` filter support in `search_by_vector` for InMemory and SQLite backends

**B — REST API Server**
- `agent_memory_manager.server` package — production-ready FastAPI service
- `create_app(manager)` factory with FastAPI lifespan (initialize/close MemoryManager)
- Endpoints: `POST /sessions/{session_id}/memories`, `GET /sessions/{session_id}/memories`,
  `POST /sessions/{session_id}/search`, `POST /sessions/{session_id}/prompt`,
  `DELETE /sessions/{session_id}`, `POST /users/{user_id}/profile`,
  `GET /users/{user_id}/profile`, `POST /users/{user_id}/search`,
  `DELETE /users/{user_id}`, `GET /sessions/{session_id}/graph/{entity}`,
  `GET /stats`
- `server/schemas.py` — Pydantic v2 request/response models
- `server/entrypoint.py` — env-var-based Docker entrypoint (`AMM_BACKEND`, `AMM_LLM_*`, etc.)
- `Dockerfile` — `python:3.12-slim`, installs `.[all]` + `fastapi` + `uvicorn`
- `docker-compose.yml` — single-service deployment on port 8000 with SQLite volumes
- `server` extras: `pip install agent-memory-manager[server]` for FastAPI + Uvicorn

**C — PgVector Backend + Streaming Compression**
- `PgVectorBackend` — PostgreSQL + pgvector production backend
  - IVFFlat ANN index (`embedding <=> $1::vector` cosine distance operator)
  - asyncpg connection pool; `ON CONFLICT DO UPDATE` upsert
  - `list_by_user` / `delete_by_user` with parameterized `$N` filters
  - Falls back to in-process cosine re-ranking after ANN pre-fetch
- `StreamingCompressStrategy` — incremental context compression
  - Stores each incoming message immediately to keep context always ready
  - When total tokens exceed `compress_threshold`, LLM-summarizes oldest messages
  - Always preserves `preserve_recent` most-recent messages verbatim
  - Graceful fallback if LLM unavailable (skips compression, keeps raw messages)
  - `build_context` uses cosine-similarity ranking; falls back to recency if no embeddings

**Package extras**
- `pgvector = ["asyncpg>=0.29.0", "pgvector>=0.3.0"]` (already present, now documented)
- `server = ["fastapi>=0.100.0", "uvicorn>=0.20.0"]` (new)
- `all` extra now includes `server`

**Tests**
- `test_user_profile.py` — 8 tests for UserProfile / UserProfileStore
- `test_manager_user.py` — 10 tests for cross-session MemoryManager methods
- `test_server.py` — 8 tests for REST API routes (ASGITransport)
- `test_streaming.py` — 10 tests for StreamingCompressStrategy (mocked backend/LLM)
- `test_pgvector.py` — 18 tests for PgVectorBackend (mocked asyncpg, no real DB)

---

## [1.5.0] — 2026-05-25

### Added

**Graph Memory (Knowledge Graph Integration)**
- `GraphExtractor` — LLM-driven entity and relation extraction from conversation turns;
  merges into per-session `SemanticMemory`; deduplicates entities (attribute merge) and
  relations (skips exact duplicates); graceful fallback on LLM failure
- `GraphStore` — async SQLite persistence for knowledge graphs; save/load/delete per
  session; `MemoryManager` auto-restores graph on first access when `graph_db_path` set
- `MemoryManager.query_graph(entity, session_id, hops)` — multi-hop neighbourhood query
- `MemoryManager.get_entity(name, session_id)` — fetch single entity details
- `MemoryManager.list_entities(session_id, entity_type)` — list entities with optional type filter
- `enable_graph` flag on `MemoryManager` (default `True`) to opt out of graph extraction
- `graph_db_path` param on `MemoryManager` to enable SQLite graph persistence
- `AddResult.entities_extracted` / `relations_extracted` — graph extraction counts
- `MemoryStats.graph_entity_count` / `graph_relation_count` — graph stats in `get_stats()`
- `GraphQueryResult` dataclass for structured query responses
- `delete_session()` now also clears the in-process graph and SQLite graph store

**Tests**
- `test_graph_extractor.py` — 7 tests covering extraction, dedup, merge, failure paths
- `test_graph_store.py` — 7 tests covering save/load/delete/overwrite/list
- `test_manager.py` — +10 graph integration tests
- Total: 118 unit tests (+22), core coverage 86%

---

## [1.0.0] — 2026-05-25

### Added

**Memory Strategies**
- `SlidingWindowStrategy` — zero-LLM-call window management; fastest option
- `SummarizeStrategy` — auto-compresses old turns via LLM summarization when token threshold exceeded
- `AtomicFactsStrategy` — two-phase Extract+Update pipeline (Mem0 architecture); extracts durable atomic facts and deduplicates against existing memories
- `ReflectionStrategy` — per-session importance accumulator; synthesizes higher-order insights when threshold exceeded (Generative Agents, Park et al. 2023)
- `ZettelkastenStrategy` — structured note creation with automatic bidirectional linking based on semantic similarity (A-MEM, arXiv:2502.12110, NeurIPS 2025)
- `StrategyPipeline` — chains multiple strategies sequentially, merges results

**Storage Backends**
- `InMemoryBackend` — zero-dependency in-process store; ideal for tests and demos
- `SQLiteBackend` — persistent lightweight backend via `aiosqlite`; production-ready for single-process deployments
- `ChromaBackend` — HNSW vector indexing via ChromaDB; supports ephemeral / persistent / remote modes
- `QdrantBackend` — production-grade Qdrant vector DB; supports in-memory / on-disk / remote server

**Memory Layers**
- `SemanticMemory` — NetworkX-based temporal knowledge graph; entity extraction, typed directed relations, multi-hop queries, JSON persistence

**LLM Clients**
- `AnthropicClient` — Anthropic Claude (claude-sonnet-4-6 default); supports prompt caching
- `OpenAIClient` — OpenAI GPT models

**Embedders**
- `OpenAIEmbedder` — text-embedding-3-small (1536 dims, default)
- `LocalEmbedder` — SentenceTransformers (offline; all-MiniLM-L6-v2 default)

**Framework Integrations**
- `AgentMemoryManagerAdapter` — LangChain `BaseMemory` drop-in adapter
- `LlamaIndexMemoryAdapter` — LlamaIndex `BaseMemory` drop-in adapter

**Infrastructure**
- `MemoryManager` — unified entry-point with `add` / `search` / `build_context` / `build_prompt` / `compress` / `delete_session` / `get_stats` + `from_config` factory
- `MemoryConfig` — Pydantic v2 config model; YAML-friendly
- Composite retrieval scoring: recency × importance × relevance (Generative Agents)
- Benchmark evaluation framework (`tests/benchmarks/eval_framework.py`)
- GitHub Actions CI matrix (Python 3.10 / 3.11 / 3.12)
- Issue templates, PR template, GitHub project setup script

### Test coverage
- 69 unit tests, 100% pass rate (2 skipped pending optional deps)
- 82% coverage of core business logic (optional-dep adapters excluded)

### Research references
- Mem0 (arXiv:2504.19413) — atomic facts extraction pipeline
- Generative Agents (Park et al., ACM UIST 2023) — reflection mechanism
- A-MEM (arXiv:2502.12110, NeurIPS 2025) — Zettelkasten dynamic linking
- LLMLingua (EMNLP 2023) — context compression
- StreamingLLM (ICLR 2024) — attention sink management
- Zep/Graphiti (arXiv:2501.13956) — temporal knowledge graph design
- HippoRAG (arXiv:2405.14831, NeurIPS 2024) — hippocampal indexing

---

## [0.1.0] — 2026-05-25

- Initial project scaffold, research docs, PRD, and technical design document
