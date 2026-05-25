from __future__ import annotations

try:
    import tiktoken

    _encoder = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_encoder.encode(text))

except ImportError:
    def count_tokens(text: str) -> int:  # type: ignore[misc]
        """Fallback: rough estimate of 4 chars per token."""
        return max(1, len(text) // 4)


def truncate_to_budget(texts: list[str], token_budget: int) -> list[str]:
    """Return as many texts as fit within the token budget (in order)."""
    result: list[str] = []
    used = 0
    for text in texts:
        tokens = count_tokens(text)
        if used + tokens > token_budget:
            break
        result.append(text)
        used += tokens
    return result
