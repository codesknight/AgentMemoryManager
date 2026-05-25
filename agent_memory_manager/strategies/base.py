from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_memory_manager.backends.base import MemoryBackend
    from agent_memory_manager.embedders.base import Embedder
    from agent_memory_manager.llm.base import LLMClient
    from agent_memory_manager.models import MemoryRecord, Message


@dataclass
class ProcessResult:
    added: list["MemoryRecord"] = field(default_factory=list)
    updated: list["MemoryRecord"] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    compressed: bool = False
    reflected: bool = False


class MemoryStrategy(ABC):
    """Base class for all memory processing strategies."""

    @abstractmethod
    async def process(
        self,
        messages: list["Message"],
        session_id: str,
        backend: "MemoryBackend",
        embedder: "Embedder",
        llm: "LLMClient",
    ) -> ProcessResult:
        """Process new messages and update the memory store."""
        ...

    @abstractmethod
    async def build_context(
        self,
        query: str,
        session_id: str,
        backend: "MemoryBackend",
        embedder: "Embedder",
        token_budget: int,
    ) -> str:
        """Return a formatted context string to inject into the prompt."""
        ...
