"""Robust JSON extraction from LLM responses.

LLMs often wrap JSON in markdown code fences or prefix it with
chain-of-thought content (<think>...</think>).  This module provides a
single helper that strips all of that and returns the first valid JSON
object or array found in the text.
"""
from __future__ import annotations

import json
import re
from typing import Any


def extract_json(text: str) -> Any:
    """Extract the first valid JSON object or array from an LLM response.

    Handles:
    - <think>...</think> reasoning blocks (Qwen3, DeepSeek-R1, etc.)
    - ```json ... ``` and ``` ... ``` markdown code fences
    - Leading/trailing prose before or after the JSON
    - Bare JSON with no wrapper

    Returns the parsed Python object.
    Raises json.JSONDecodeError if no valid JSON is found.
    """
    # 1. Strip <think>...</think> blocks (greedy=False to handle multiple)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # 2. Extract content from markdown code fences
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1)

    text = text.strip()

    # 3. Try direct parse first (common fast path)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 4. Find the first JSON array or object by scanning for [ or {
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        start = text.find(start_char)
        if start == -1:
            continue
        # Walk forward matching brackets to find the closing character
        depth = 0
        in_string = False
        escape_next = False
        for i, ch in enumerate(text[start:], start):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break

    raise json.JSONDecodeError("No valid JSON found in response", text, 0)
