from __future__ import annotations

from .base import Embedder

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:
    raise ImportError(
        "Install 'sentence-transformers' to use LocalEmbedder: "
        "pip install sentence-transformers"
    ) from exc


class LocalEmbedder(Embedder):
    """Offline text embedder using SentenceTransformers.

    No API key needed. Recommended for air-gapped or cost-sensitive deployments.
    Default model: all-MiniLM-L6-v2 (384 dims, fast, good quality for retrieval).
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        self._model = SentenceTransformer(model)
        self._dimensions = self._model.get_sentence_embedding_dimension() or 384

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, text: str) -> list[float]:
        vector = self._model.encode(text, convert_to_numpy=True)
        return vector.tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(texts, convert_to_numpy=True)
        return [v.tolist() for v in vectors]
