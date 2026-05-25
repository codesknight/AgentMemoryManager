from __future__ import annotations

from typing import Optional

from .base import Embedder

try:
    from openai import AsyncOpenAI
except ImportError as exc:
    raise ImportError("Install 'openai' to use OpenAIEmbedder: pip install openai") from exc

_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder(Embedder):
    """Text embedder using OpenAI embedding models.

    text-embedding-3-small is the recommended default: good quality,
    low cost, and 1536 dimensions compatible with most vector databases.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model
        self._client = AsyncOpenAI(api_key=api_key)

    @property
    def dimensions(self) -> int:
        return _DIMENSIONS.get(self.model, 1536)

    async def embed(self, text: str) -> list[float]:
        response = await self._client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await self._client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
