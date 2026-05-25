"""Unit tests for data models."""
import pytest
from datetime import datetime, timezone

from agent_memory_manager.models import Message, MemoryRecord, MemoryType, Role


def test_message_defaults():
    msg = Message(role=Role.USER, content="Hello")
    assert msg.id
    assert msg.role == Role.USER
    assert msg.content == "Hello"
    assert isinstance(msg.created_at, datetime)


def test_message_token_estimate():
    msg = Message(role=Role.USER, content="a" * 400)
    assert msg.token_estimate() == 101


def test_memory_record_defaults():
    record = MemoryRecord(session_id="s1", content="The user likes Python")
    assert record.id
    assert record.memory_type == MemoryType.EPISODIC
    assert record.importance_score == 5.0
    assert record.links == []


def test_memory_record_touch():
    record = MemoryRecord(session_id="s1", content="fact")
    original = record.accessed_at
    record.touch()
    assert record.accessed_at >= original
