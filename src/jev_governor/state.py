"""Projecting an agent transcript into bounded state for one decision.

The governor re-asks on every step, so the *size* of what it sends is the whole cost model. A
projection is built fresh each time and thrown away; the agent's canonical transcript is never
touched, truncated, or reordered. That separation matters more than it looks: the transcript is
what the generative model and the user's own history depend on, and a decision layer that
rewrites it to save tokens has corrupted the record to make its own job cheaper.

What a projection keeps, and why that shape:

* ``goal`` -- the most recent user message, because the question is always "what next for
  *this* request", not for the whole session.
* ``instructions`` -- the system messages, concatenated once in original order. The agent's
  identity is durable context; repeating it per-row would waste most of the budget.
* ``recent_history`` -- the newest turns that fit, oldest-first, walked backwards from the end
  so the newest evidence is the content that survives a tight budget.
* ``truncated`` -- whether anything was dropped. The governor reads this flag and refuses to
  accept a ``finish`` built on a partial view, which is the single most valuable thing in this
  module.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

#: Share of the budget reserved for the current goal and for the system instructions.
_GOAL_SHARE = 4
_INSTRUCTION_SHARE = 2


def text_of(content: Any) -> str:
    """Flatten a message body to plain text.

    Handles the bare-string form and the OpenAI content-parts list. Non-text parts (images,
    audio) are intentionally dropped rather than described: System One reads text, and an
    invented placeholder like ``[image]`` is a claim about content nobody verified.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _size(state: dict) -> int:
    return len(json.dumps(state, ensure_ascii=False))


def project(messages: list[dict], limit: int) -> dict:
    """Return a bounded projection of ``messages`` whose serialized size is at most ``limit``.

    ``limit`` is enforced against the *serialized* length, not the sum of field lengths, because
    JSON escaping can multiply a string several times over -- a transcript full of control
    characters or emoji overruns a naive character count badly.
    """
    goal_budget = max(1, limit // _GOAL_SHARE)
    instruction_budget = max(1, limit // _INSTRUCTION_SHARE)

    goal_full = next(
        (text_of(m.get("content")) for m in reversed(messages) if m.get("role") == "user"), ""
    )
    instructions_full = "\n\n".join(
        text_of(m.get("content")) for m in messages if m.get("role") == "system"
    )

    state: dict[str, Any] = {
        "goal": goal_full[:goal_budget],
        "instructions": instructions_full[:instruction_budget],
        "recent_history": [],
        "truncated": len(goal_full) > goal_budget or len(instructions_full) > instruction_budget,
    }

    for message in reversed(messages):
        if message.get("role") == "system":
            continue  # already carried once, in order, by `instructions`
        body = text_of(message.get("content"))
        row: dict[str, Any] = {"role": message.get("role", ""), "content": body[:goal_budget]}
        if message.get("tool_calls"):
            row["tool_calls"] = deepcopy(message["tool_calls"])
        if message.get("name"):
            row["name"] = message["name"]

        state["recent_history"].insert(0, row)
        if _size(state) > limit:
            state["recent_history"].pop(0)
            state["truncated"] = True
            break
        if len(body) > goal_budget:
            state["truncated"] = True

    # Escaping can still push a just-fitting projection over the line. Shrink the two
    # variable-length text fields until it fits, then -- if even an empty-text projection is
    # too large -- drop history rows. Each branch strictly reduces size, so this terminates.
    while _size(state) > limit:
        if state["goal"] or state["instructions"]:
            state["goal"] = state["goal"][: len(state["goal"]) // 2]
            state["instructions"] = state["instructions"][: len(state["instructions"]) // 2]
        elif state["recent_history"]:
            state["recent_history"].pop(0)
        else:
            break
        state["truncated"] = True

    return state
