"""Unit tests for token_counter utilities."""
import pytest

from agent_memory_manager.utils.token_counter import count_tokens, truncate_to_budget


def test_count_tokens_non_empty():
    n = count_tokens("Hello, world!")
    assert isinstance(n, int)
    assert n > 0


def test_count_tokens_empty():
    # Empty string should return at least 1 (due to +1 in fallback) or 0
    n = count_tokens("")
    assert isinstance(n, int)
    assert n >= 0


def test_count_tokens_longer_text():
    short = count_tokens("hi")
    long = count_tokens("This is a much longer sentence with many more words in it.")
    assert long > short


def test_truncate_to_budget_all_fit():
    texts = ["hello", "world", "foo"]
    result = truncate_to_budget(texts, token_budget=1000)
    assert result == texts


def test_truncate_to_budget_limits():
    texts = ["word " * 100, "word " * 100, "word " * 100]
    result = truncate_to_budget(texts, token_budget=50)
    # Should keep only as many texts as fit within 50 tokens
    assert len(result) < len(texts)


def test_truncate_to_budget_empty_input():
    assert truncate_to_budget([], token_budget=100) == []


def test_truncate_to_budget_zero_budget():
    # No text should fit in a 0-token budget
    result = truncate_to_budget(["hello"], token_budget=0)
    assert result == []
