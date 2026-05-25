from __future__ import annotations

import asyncio
from typing import Any

try:
    from langchain_core.memory import BaseMemory
    from langchain_core.messages import AIMessage, HumanMessage
except ImportError as exc:
    raise ImportError(
        "Install 'langchain-core' to use AgentMemoryManagerAdapter: "
        "pip install langchain-core"
    ) from exc

from agent_memory_manager.manager import MemoryManager
from agent_memory_manager.models import Message, Role


def _get_event_loop() -> asyncio.AbstractEventLoop:
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


class AgentMemoryManagerAdapter(BaseMemory):
    """Adapts AgentMemoryManager as a LangChain BaseMemory drop-in.

    Usage::

        memory = AgentMemoryManagerAdapter(
            manager=manager,
            session_id="user-123",
        )
        chain = ConversationChain(llm=llm, memory=memory)
    """

    manager: Any  # MemoryManager — typed as Any to satisfy Pydantic v2
    session_id: str
    token_budget: int = 2000
    memory_key: str = "history"

    class Config:
        arbitrary_types_allowed = True

    @property
    def memory_variables(self) -> list[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        query = inputs.get("input") or inputs.get("human_input", "")
        loop = _get_event_loop()
        ctx = loop.run_until_complete(
            self.manager.build_context(
                query=query,
                session_id=self.session_id,
                token_budget=self.token_budget,
            )
        )
        return {self.memory_key: ctx.context}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
        user_content = inputs.get("input") or inputs.get("human_input", "")
        ai_content = outputs.get("response") or outputs.get("output", "")
        messages = []
        if user_content:
            messages.append(Message(role=Role.USER, content=user_content))
        if ai_content:
            messages.append(Message(role=Role.ASSISTANT, content=ai_content))
        if not messages:
            return
        loop = _get_event_loop()
        loop.run_until_complete(
            self.manager.add(messages=messages, session_id=self.session_id)
        )

    def clear(self) -> None:
        loop = _get_event_loop()
        loop.run_until_complete(self.manager.delete_session(self.session_id))
