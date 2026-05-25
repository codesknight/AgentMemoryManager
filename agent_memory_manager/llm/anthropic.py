from __future__ import annotations

from typing import Optional

from .base import LLMClient

try:
    import anthropic as _anthropic
except ImportError as exc:
    raise ImportError("Install 'anthropic' to use AnthropicClient: pip install anthropic") from exc


class AnthropicClient(LLMClient):
    """LLM client backed by Anthropic Claude.

    Recommended for memory operations (summarize, extract, reflect) because
    Claude's instruction-following is precise and supports prompt caching,
    which cuts latency and cost on repeated system-prompt prefixes.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: Optional[str] = None,
        max_retries: int = 2,
    ) -> None:
        self.model = model
        self._client = _anthropic.AsyncAnthropic(
            api_key=api_key,
            max_retries=max_retries,
        )

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        kwargs: dict = dict(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        if system:
            kwargs["system"] = system

        message = await self._client.messages.create(**kwargs)
        return message.content[0].text  # type: ignore[index]

    async def count_tokens(self, text: str) -> int:
        response = await self._client.messages.count_tokens(
            model=self.model,
            messages=[{"role": "user", "content": text}],
        )
        return response.input_tokens
