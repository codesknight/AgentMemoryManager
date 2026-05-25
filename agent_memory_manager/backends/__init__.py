from .base import MemoryBackend
from .in_memory import InMemoryBackend
from .sqlite import SQLiteBackend


def __getattr__(name: str):
    if name == "ChromaBackend":
        from .chroma import ChromaBackend  # noqa: PLC0415
        return ChromaBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["MemoryBackend", "InMemoryBackend", "SQLiteBackend", "ChromaBackend"]
