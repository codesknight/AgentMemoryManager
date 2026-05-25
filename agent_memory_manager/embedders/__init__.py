from .base import Embedder
from .openai_embedder import OpenAIEmbedder

# LocalEmbedder requires 'sentence-transformers' (optional dep).
# Import it lazily to avoid breaking users who haven't installed it.
def __getattr__(name: str):
    if name == "LocalEmbedder":
        from .local_embedder import LocalEmbedder  # noqa: PLC0415
        return LocalEmbedder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["Embedder", "OpenAIEmbedder", "LocalEmbedder"]
