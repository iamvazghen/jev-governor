"""Shared fakes. No network, no API key, no provider -- the suite must run on a laptop offline.

A test suite that needs a paid credential is a test suite that does not run in CI, and one that
does not run in CI is decoration. Live verification is a separate, explicitly-run script:
``scripts/verify_live.py``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest


def answer_choice(selected: str, mass: dict[str, float], confidence: float | None = None) -> dict:
    """Build a well-formed ``choice`` answer. ``confidence`` defaults to the winner's mass."""
    return {
        "type": "choice",
        "choice": selected,
        "probabilities": dict(mass),
        "confidence": mass[selected] if confidence is None else confidence,
    }


def answer_noul(probability: float) -> dict:
    return {"type": "noul", "noul": probability}


def spread(options: set[str], selected: str, selected_mass: float) -> dict[str, float]:
    """Spread the remaining mass evenly over the other options, so it sums to one."""
    others = sorted(options - {selected})
    if not others:
        return {selected: 1.0}
    share = (1.0 - selected_mass) / len(others)
    mass = {name: share for name in others}
    mass[selected] = selected_mass
    return mass


#: The ballot a single-tool host produces: three controls plus one tool.
DEFAULT_OPTIONS = frozenset({"finish", "clarify", "delegate", "tool:read_file"})


def body(
    *,
    action: str = "tool:read_file",
    action_mass: dict[str, float] | None = None,
    goal: float = 0.1,
    risk: str = "low",
    risk_mass: dict[str, float] | None = None,
    model: str = "jev-1.13.0",
    options: frozenset[str] = DEFAULT_OPTIONS,
) -> dict:
    """A complete, valid System One response body.

    The default distribution covers exactly ``options`` and puts ``action`` at its argmax,
    because that is what a real answer looks like and the parser insists on it.
    """
    if action_mass is None:
        action_mass = spread(set(options), action, 0.95)
    if risk_mass is None:
        risk_mass = {"low": 0.0, "medium": 0.0, "high": 0.0}
        risk_mass[risk] = 1.0
    return {
        "model": model,
        "usage": {"input_tokens": 120, "output_tokens": 4},
        "answers": {
            "action": answer_choice(action, action_mass),
            "goal_met": answer_noul(goal),
            "risk": answer_choice(risk, risk_mass),
        },
    }


class FakeResponse:
    def __init__(self, status: int, payload: Any, *, raw: str | None = None) -> None:
        self.status_code = status
        self._payload = payload
        self._raw = raw

    def json(self) -> Any:
        if self._raw is not None:
            return json.loads(self._raw)
        return self._payload


class FakeTransport:
    """Records every request and returns queued responses."""

    def __init__(self, *responses: FakeResponse) -> None:
        self._queue = list(responses)
        self.requests: list[dict] = []
        self.closed = False

    def post(self, url: str, *, headers: dict, json: dict, timeout: float) -> FakeResponse:
        self.requests.append(
            {"url": url, "headers": headers, "payload": json, "timeout": timeout}
        )
        if not self._queue:
            raise AssertionError("FakeTransport ran out of queued responses")
        return self._queue.pop(0)

    def close(self) -> None:
        self.closed = True


class FakeHost:
    """A :class:`jev_governor.host.Host` that records what the governor asked it to do."""

    def __init__(
        self,
        *,
        tools: list[dict] | None = None,
        messages: list[dict] | None = None,
        tool_calls: list[str] | None = None,
        interrupted: bool = False,
    ) -> None:
        self._tools = tools if tools is not None else [_tool("read_file", "Read a file.")]
        self._messages = messages if messages is not None else [
            {"role": "system", "content": "You are a careful assistant."},
            {"role": "user", "content": "What is in notes.txt?"},
        ]
        self._tool_calls = tool_calls
        self._interrupted = interrupted
        self.generated: list[dict] = []
        self.said: list[str] = []
        self.static_calls: list[tuple[str, dict]] = []

    def tools(self) -> list[dict]:
        return self._tools

    def messages(self) -> list[dict]:
        return self._messages

    def interrupted(self) -> bool:
        return self._interrupted

    def generate(self, *, force_tool: str | None, directive: str) -> Any:
        self.generated.append({"force_tool": force_tool, "directive": directive})
        return {"kind": "generated", "force_tool": force_tool}

    def called_tools(self, response: Any) -> list[str]:
        if self._tool_calls is not None:
            return list(self._tool_calls)
        forced = response.get("force_tool") if isinstance(response, dict) else None
        return [forced] if forced else []

    def say(self, text: str) -> Any:
        self.said.append(text)
        return {"kind": "said", "text": text}

    def say_tool_call(self, name: str, arguments: dict) -> Any:
        self.static_calls.append((name, arguments))
        return {"kind": "static", "name": name, "arguments": arguments}


def _tool(name: str, description: str, parameters: dict | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters
            or {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    }


@pytest.fixture
def tool():
    return _tool
