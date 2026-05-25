from __future__ import annotations

from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Abstract interface for all LLM providers."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str: ...

    @abstractmethod
    async def count_tokens(self, text: str) -> int: ...
