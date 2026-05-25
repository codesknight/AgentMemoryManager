from .base import Embedder
from .openai_embedder import OpenAIEmbedder

# LocalEmbedder requires 'sentence-transformers' (optional dep).
# OllamaEmbedder requires 'httpx' (optional dep).
# Both are loaded lazily to avoid import errors when deps are absent.
def __getattr__(name: str):
    if name == "LocalEmbedder":
        from .local_embedder import LocalEmbedder  # noqa: PLC0415
        return LocalEmbedder
    if name == "OllamaEmbedder":
        from .ollama_embedder import OllamaEmbedder  # noqa: PLC0415
        return OllamaEmbedder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["Embedder", "OpenAIEmbedder", "LocalEmbedder", "OllamaEmbedder"]
