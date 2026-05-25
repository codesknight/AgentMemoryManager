from __future__ import annotations

import asyncio
from typing import Any, Optional

try:
    from llama_index.core.memory import BaseMemory
    from llama_index.core.llms import ChatMessage, MessageRole
except ImportError as exc:
    raise ImportError(
        "Install 'llama-index-core' to use LlamaIndexMemoryAdapter: "
        "pip install llama-index-core"
    ) from exc

from agent_memory_manager.manager import MemoryManager
from agent_memory_manager.models import Message, Role


def _get_loop() -> asyncio.AbstractEventLoop:
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


def _to_internal(msg: ChatMessage) -> Message:
    role = Role.USER if msg.role == MessageRole.USER else Role.ASSISTANT
    return Message(role=role, content=str(msg.content))


class LlamaIndexMemoryAdapter(BaseMemory):
    """Adapts AgentMemoryManager as a LlamaIndex BaseMemory drop-in.

    Usage::

        from llama_index.core import Settings
        from llama_index.core.chat_engine import CondensePlusContextChatEngine

        memory = LlamaIndexMemoryAdapter(
            manager=manager,
            session_id="user-123",
            token_budget=2000,
        )
    """

    def __init__(
        self,
        manager: MemoryManager,
        session_id: str,
        token_budget: int = 2000,
    ) -> None:
        super().__init__()
        self._manager = manager
        self._session_id = session_id
        self._token_budget = token_budget
        self._buffer: list[ChatMessage] = []

    @classmethod
    def from_defaults(
        cls,
        manager: MemoryManager,
        session_id: str,
        token_budget: int = 2000,
    ) -> "LlamaIndexMemoryAdapter":
        return cls(manager=manager, session_id=session_id, token_budget=token_budget)

    def get(self, input: Optional[str] = None, **kwargs: Any) -> list[ChatMessage]:
        """Return recent messages enriched with memory context."""
        query = input or ""
        loop = _get_loop()
        ctx = loop.run_until_complete(
            self._manager.build_context(
                query=query,
                session_id=self._session_id,
                token_budget=self._token_budget,
            )
        )
        result = list(self._buffer)
        if ctx.context.strip():
            # Prepend memory as a system message
            result.insert(
                0,
                ChatMessage(
                    role=MessageRole.SYSTEM,
                    content=f"Relevant memory from past conversations:\n{ctx.context}",
                ),
            )
        return result

    def get_all(self) -> list[ChatMessage]:
        return list(self._buffer)

    def put(self, message: ChatMessage) -> None:
        """Persist a new message into memory."""
        self._buffer.append(message)
        internal = _to_internal(message)
        loop = _get_loop()
        loop.run_until_complete(
            self._manager.add(messages=[internal], session_id=self._session_id)
        )

    def set(self, messages: list[ChatMessage]) -> None:
        self._buffer = list(messages)

    def reset(self) -> None:
        self._buffer.clear()
        loop = _get_loop()
        loop.run_until_complete(self._manager.delete_session(self._session_id))
