from __future__ import annotations

from typing import Optional

from .base import LLMClient

try:
    from openai import AsyncOpenAI
except ImportError as exc:
    raise ImportError("Install 'openai' to use OpenAIClient: pip install openai") from exc


class OpenAIClient(LLMClient):
    """LLM client backed by OpenAI GPT models."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model
        self._client = AsyncOpenAI(api_key=api_key)

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    async def count_tokens(self, text: str) -> int:
        from agent_memory_manager.utils.token_counter import count_tokens
        return count_tokens(text)
