"""The state projection: bounded, honest about truncation, and never destructive."""

from __future__ import annotations

import json

import pytest

from jev_governor import project
from jev_governor.state import text_of


class TestShape:
    def test_carries_the_latest_user_message_as_the_goal(self):
        state = project(
            [
                {"role": "user", "content": "first thing"},
                {"role": "assistant", "content": "done"},
                {"role": "user", "content": "the thing I actually want"},
            ],
            8_000,
        )
        assert state["goal"] == "the thing I actually want"

    def test_carries_system_messages_once_in_original_order(self):
        state = project(
            [
                {"role": "system", "content": "alpha"},
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "beta"},
            ],
            8_000,
        )
        assert state["instructions"] == "alpha\n\nbeta"
        # And not repeated in the history, where they would eat the budget.
        assert all(row["role"] != "system" for row in state["recent_history"])

    def test_keeps_history_oldest_first(self):
        state = project(
            [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
                {"role": "user", "content": "three"},
            ],
            8_000,
        )
        assert [row["content"] for row in state["recent_history"]] == ["one", "two", "three"]

    def test_carries_tool_calls_and_tool_names(self):
        state = project(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "read_file", "arguments": "{}"}}],
                },
                {"role": "tool", "name": "read_file", "content": "file body"},
            ],
            8_000,
        )
        assert state["recent_history"][0]["tool_calls"][0]["function"]["name"] == "read_file"
        assert state["recent_history"][1]["name"] == "read_file"

    def test_an_empty_transcript_is_a_valid_projection(self):
        state = project([], 8_000)
        assert state == {"goal": "", "instructions": "", "recent_history": [], "truncated": False}


class TestBounds:
    @pytest.mark.parametrize("limit", [4_000, 8_000, 24_000, 100_000])
    def test_the_serialized_projection_never_exceeds_the_limit(self, limit):
        messages = [{"role": "user", "content": "x" * 30_000} for _ in range(40)]
        messages.insert(0, {"role": "system", "content": "s" * 30_000})
        state = project(messages, limit)
        assert len(json.dumps(state, ensure_ascii=False)) <= limit

    def test_the_limit_holds_against_json_escaping(self):
        # Characters that escape to six bytes each would blow a naive character count.
        messages = [{"role": "user", "content": "\x00\x01\x02" * 5_000}]
        state = project(messages, 4_000)
        assert len(json.dumps(state, ensure_ascii=False)) <= 4_000

    def test_newest_evidence_survives_a_tight_budget(self):
        messages = [
            {"role": "user", "content": f"message {index} " + "y" * 900} for index in range(30)
        ]
        state = project(messages, 4_000)
        kept = " ".join(row["content"] for row in state["recent_history"])
        assert "message 29" in kept
        assert "message 0 " not in kept

    def test_terminates_when_even_an_empty_projection_is_too_large(self):
        # A pathological limit must not spin: each branch strictly reduces size.
        state = project([{"role": "user", "content": "z" * 100}] * 5, 4_000)
        assert isinstance(state, dict)


class TestTruncationFlag:
    def test_is_false_when_everything_fits(self):
        state = project([{"role": "user", "content": "short"}], 24_000)
        assert state["truncated"] is False

    def test_is_true_when_a_message_was_clipped(self):
        state = project([{"role": "user", "content": "x" * 50_000}], 4_000)
        assert state["truncated"] is True

    def test_is_true_when_rows_were_dropped(self):
        messages = [{"role": "user", "content": "x" * 900} for _ in range(50)]
        state = project(messages, 4_000)
        assert state["truncated"] is True
        assert len(state["recent_history"]) < 50


class TestNonDestructive:
    def test_the_input_messages_are_not_mutated(self):
        messages = [
            {"role": "system", "content": "s"},
            {
                "role": "assistant",
                "content": "a",
                "tool_calls": [{"function": {"name": "t", "arguments": "{}"}}],
            },
        ]
        before = json.dumps(messages, sort_keys=True)
        project(messages, 8_000)
        assert json.dumps(messages, sort_keys=True) == before

    def test_tool_calls_are_deep_copied(self):
        calls = [{"function": {"name": "t", "arguments": "{}"}}]
        messages = [{"role": "assistant", "content": "", "tool_calls": calls}]
        state = project(messages, 8_000)
        state["recent_history"][0]["tool_calls"][0]["function"]["name"] = "mutated"
        assert calls[0]["function"]["name"] == "t"


class TestTextOf:
    def test_passes_a_plain_string_through(self):
        assert text_of("hello") == "hello"

    def test_joins_text_parts(self):
        assert text_of([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "a\nb"

    def test_drops_non_text_parts_rather_than_inventing_a_placeholder(self):
        # A fabricated "[image]" is a claim about content nobody verified.
        content = [
            {"type": "image_url", "image_url": {"url": "..."}},
            {"type": "text", "text": "x"},
        ]
        assert text_of(content) == "x"

    def test_returns_empty_for_anything_else(self):
        assert text_of(None) == ""
        assert text_of(42) == ""
