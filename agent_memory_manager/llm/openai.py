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
        keep_alive: Optional[int] = None,
        trust_env: bool = False,
    ) -> None:
        self.model = model
        # keep_alive controls how long Ollama keeps the model in VRAM after a call.
        # Set to 0 to release immediately (recommended when sharing GPU with an embedder).
        self._keep_alive = keep_alive
        # trust_env=False bypasses system/WinINet proxy — correct for local Ollama.
        # Set trust_env=True for external APIs (OpenAI, Doubao, etc.) that need system proxy.
        http_client = httpx.AsyncClient(trust_env=trust_env, timeout=timeout)
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

        extra: dict = {}
        if self._keep_alive is not None:
            extra["keep_alive"] = self._keep_alive

        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=extra or None,
        )
        return response.choices[0].message.content or ""

    async def count_tokens(self, text: str) -> int:
        from agent_memory_manager.utils.token_counter import count_tokens
        return count_tokens(text)
