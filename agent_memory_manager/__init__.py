"""AgentMemoryManager — Pluggable memory management for LLM agents."""

from .config import MemoryConfig
from .manager import MemoryManager
from .models import Message, MemoryRecord, MemoryType, Role

__version__ = "0.1.0"

__all__ = [
    "MemoryConfig",
    "MemoryManager",
    "Message",
    "MemoryRecord",
    "MemoryType",
    "Role",
]
