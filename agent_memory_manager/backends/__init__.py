from .base import MemoryBackend
from .in_memory import InMemoryBackend
from .sqlite import SQLiteBackend


def __getattr__(name: str):
    if name == "ChromaBackend":
        from .chroma import ChromaBackend  # noqa: PLC0415
        return ChromaBackend
    if name == "QdrantBackend":
        from .qdrant import QdrantBackend  # noqa: PLC0415
        return QdrantBackend
    if name == "PgVectorBackend":
        from .pgvector import PgVectorBackend  # noqa: PLC0415
        return PgVectorBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MemoryBackend",
    "InMemoryBackend",
    "SQLiteBackend",
    "ChromaBackend",
    "QdrantBackend",
    "PgVectorBackend",
]
