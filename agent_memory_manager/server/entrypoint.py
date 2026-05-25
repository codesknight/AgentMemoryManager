"""Default server entrypoint — configures MemoryManager from environment variables.

Environment variables:
    AMM_BACKEND        sqlite | in_memory (default: sqlite)
    AMM_DB_PATH        path to SQLite file (default: /data/memory.db)
    AMM_GRAPH_DB_PATH  path to graph SQLite file (default: /data/graph.db)
    AMM_PROFILE_DB_PATH path to profile SQLite file (default: /data/profiles.db)
    AMM_LLM_PROVIDER   anthropic | openai (default: openai)
    AMM_LLM_MODEL      model name (default: gpt-4o-mini)
    AMM_LLM_API_KEY    API key
    AMM_LLM_BASE_URL   custom base URL (e.g. Ollama)
    AMM_EMBED_MODEL    openai | local (default: local)
"""
import os

from agent_memory_manager.backends.sqlite import SQLiteBackend
from agent_memory_manager.backends.in_memory import InMemoryBackend
from agent_memory_manager.embedders.local_embedder import LocalEmbedder
from agent_memory_manager.llm.openai import OpenAIClient
from agent_memory_manager.manager import MemoryManager
from agent_memory_manager.strategies.atomic_facts import AtomicFactsStrategy

from .app import create_app

_backend_type = os.environ.get("AMM_BACKEND", "sqlite")
_db_path = os.environ.get("AMM_DB_PATH", "/data/memory.db")
_graph_db = os.environ.get("AMM_GRAPH_DB_PATH", "/data/graph.db")
_profile_db = os.environ.get("AMM_PROFILE_DB_PATH", "/data/profiles.db")

backend = SQLiteBackend(_db_path) if _backend_type == "sqlite" else InMemoryBackend()

llm = OpenAIClient(
    model=os.environ.get("AMM_LLM_MODEL", "gpt-4o-mini"),
    api_key=os.environ.get("AMM_LLM_API_KEY"),
    base_url=os.environ.get("AMM_LLM_BASE_URL"),
    trust_env=True,
)

embedder = LocalEmbedder(model=os.environ.get("AMM_EMBED_MODEL", "all-MiniLM-L6-v2"))

manager = MemoryManager(
    backend=backend,
    strategy=AtomicFactsStrategy(),
    llm=llm,
    embedder=embedder,
    enable_graph=True,
    graph_db_path=_graph_db,
    user_profile_db_path=_profile_db,
)

app = create_app(manager)
