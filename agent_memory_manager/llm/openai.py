from __future__ import annotations

from typing import Optional

from .base import LLMClient

try:
    from openai import AsyncOpenAI
    import httpx
except ImportError as exc:
    raise ImportError("Install 'openai' to use OpenAIClient: pip install openai") from exc


class OpenAIClient(LLMClient):
    """LLM client backed by OpenAI GPT models."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 10,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        # trust_env=False bypasses system/WinINet proxy for local endpoints
        http_client = httpx.AsyncClient(trust_env=False, timeout=timeout)
        self._client = AsyncOpenAI(
            api_key=api_key or "ollama",
            base_url=base_url,
            max_retries=max_retries,
            timeout=timeout,
            http_client=http_client,
        )

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
            extra_body={"keep_alive": 0},  # release model from VRAM immediately
        )
        return response.choices[0].message.content or ""

    async def count_tokens(self, text: str) -> int:
        from agent_memory_manager.utils.token_counter import count_tokens
        return count_tokens(text)
