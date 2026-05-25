from __future__ import annotations

from abc import ABC, abstractmethod


class Embedder(ABC):
    """Abstract interface for text embedding providers."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]: ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...

    @property
    @abstractmethod
    def dimensions(self) -> int: ...
