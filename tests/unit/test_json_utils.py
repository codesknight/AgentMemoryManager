"""Unit tests for extract_json utility."""
import json
import pytest

from agent_memory_manager.utils.json_utils import extract_json


def test_plain_json_array():
    assert extract_json('[{"fact": "x", "importance": 7}]') == [{"fact": "x", "importance": 7}]


def test_plain_json_object():
    assert extract_json('{"action": "add", "target_id": null}') == {"action": "add", "target_id": None}


def test_markdown_code_fence_json():
    text = '```json\n[{"fact": "hello", "importance": 5}]\n```'
    assert extract_json(text) == [{"fact": "hello", "importance": 5}]


def test_markdown_code_fence_no_lang():
    text = '```\n{"action": "skip", "target_id": null}\n```'
    assert extract_json(text) == {"action": "skip", "target_id": None}


def test_think_tag_stripped():
    text = '<think>Let me reason about this...</think>\n[{"fact": "user is Alice", "importance": 8}]'
    assert extract_json(text) == [{"fact": "user is Alice", "importance": 8}]


def test_think_tag_with_code_fence():
    text = '<think>reasoning</think>\n```json\n{"action": "add", "target_id": null}\n```'
    assert extract_json(text) == {"action": "add", "target_id": None}


def test_json_embedded_in_prose():
    text = 'Here is my answer:\n[{"fact": "embedded", "importance": 6}]\nThat is all.'
    assert extract_json(text) == [{"fact": "embedded", "importance": 6}]


def test_empty_array():
    assert extract_json("[]") == []


def test_nested_object():
    data = '{"a": {"b": [1, 2, 3]}, "c": true}'
    assert extract_json(data) == {"a": {"b": [1, 2, 3]}, "c": True}


def test_raises_on_no_json():
    with pytest.raises(json.JSONDecodeError):
        extract_json("no json here at all")


def test_raises_on_empty_string():
    with pytest.raises(json.JSONDecodeError):
        extract_json("")


def test_multiple_think_blocks_stripped():
    text = "<think>first</think> some text <think>second</think> [1, 2, 3]"
    assert extract_json(text) == [1, 2, 3]
