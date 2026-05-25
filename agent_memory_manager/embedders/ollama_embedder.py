from __future__ import annotations

import json
from typing import Optional

try:
    import httpx
except ImportError as exc:
    raise ImportError("Install 'httpx' to use OllamaEmbedder: pip install httpx") from exc

from .base import Embedder

# Ollama embedding model → output dimensions
_KNOWN_DIMS: dict[str, int] = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
    "llama3.2": 3072,
    "llama3.1": 4096,
    "llama3": 4096,
    "mistral": 4096,
    "qwen2": 3584,
}


class OllamaEmbedder(Embedder):
    """Embedder backed by the local Ollama /api/embeddings endpoint.

    Ollama serves embeddings for any pulled model. For dedicated embedding
    quality, prefer a model like ``nomic-embed-text``; generative models
    also work and produce embeddings of their hidden-state dimension.
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        dimensions: Optional[int] = None,
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._dims: Optional[int] = dimensions or _KNOWN_DIMS.get(model)

    # ── public API ────────────────────────────────────────────────────────────

    async def embed(self, text: str) -> list[float]:
        vec = await self._call(text)
        if self._dims is None:
            self._dims = len(vec)
        return vec

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            results.append(await self.embed(text))
        return results

    @property
    def dimensions(self) -> int:
        if self._dims is None:
            raise RuntimeError(
                "Dimensions unknown — call embed() at least once first, "
                "or pass dimensions= in the constructor."
            )
        return self._dims

    # ── internal ──────────────────────────────────────────────────────────────

    async def _call(self, text: str) -> list[float]:
        import asyncio
        # trust_env=False bypasses system proxy (e.g. Clash/V2Ray on Windows)
        for attempt in range(10):
            async with httpx.AsyncClient(timeout=self._timeout, trust_env=False) as client:
                # Try newer /api/embed endpoint first, fall back to /api/embeddings
                for endpoint, payload in [
                    (f"{self._base_url}/api/embed", {"model": self.model, "input": text}),
                    (f"{self._base_url}/api/embeddings", {"model": self.model, "prompt": text}),
                ]:
                    resp = await client.post(endpoint, json=payload)
                    if resp.status_code == 503:
                        break  # retry outer loop
                    if resp.status_code == 404:
                        continue  # try next endpoint
                    resp.raise_for_status()
                    data = resp.json()
                    # /api/embed returns {"embeddings": [[...]]}
                    # /api/embeddings returns {"embedding": [...]}
                    if "embeddings" in data:
                        return data["embeddings"][0]
                    return data["embedding"]
                wait = min(2 ** attempt, 30)
                await asyncio.sleep(wait)
        raise RuntimeError("Ollama returned 503 after 10 retries — model may be unavailable")
