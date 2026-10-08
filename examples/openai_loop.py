#!/usr/bin/env python
"""A complete governed agent loop in about fifty lines.

    TYPESAFE_API_KEY=... OPENAI_API_KEY=... python examples/openai_loop.py

This is the shape to copy if you are not using Hermes or Pydantic AI: build a request, let the
governor route it, execute whatever tool came back, append the result, repeat. The governor
replaces the line where you would have called the provider -- nothing else about the loop
changes.

Note what is *not* here: no prompt asking the model to pick carefully, no retry on a
mis-chosen tool, no parsing a tool name out of prose. The decision arrives typed, and
`Governor` has already refused to execute anything the provider substituted for it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jev_governor import Governor, GovernorConfig, JevClient, JevError
from jev_governor.adapters.openai_chat import OpenAIChatHost

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": (
                "Run a read-only shell command on the local host and return its stdout. "
                "Use for inspecting processes, ports, disk usage and file contents."
            ),
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
]


def run_shell(command: str) -> str:
    """Deliberately not a sandbox. An example, not a safe executor -- see the note below."""
    try:
        done = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30, check=False
        )
    except subprocess.TimeoutExpired:
        return "ERROR: the command timed out"
    return done.stdout if done.returncode == 0 else f"ERROR: {done.stderr[:500]}"


def main() -> int:
    key = os.environ.get("TYPESAFE_API_KEY", "")
    if not key:
        print("Set TYPESAFE_API_KEY.")
        return 2
    try:
        from openai import OpenAI
    except ImportError:
        print("This example needs the openai package: pip install openai")
        return 2

    provider = OpenAI()
    governor = Governor(
        JevClient(key),
        GovernorConfig(
            confidence_threshold=0.80,
            on_error="delegate",
            goal_hint="You administer a single Linux workstation.",
            # `static_tools={"docker_ps": {}}` would make a fixed-argument tool cost nothing at
            # all -- no generative call. Not used here: `run_shell` needs a real command
            # written each time, and pinning one would hardcode the answer this example is
            # supposed to work out.
        ),
    )

    task = " ".join(sys.argv[1:]) or "Is anything listening on port 8080? Report what you find."
    messages: list[dict] = [
        {"role": "system", "content": "You are a careful operations assistant."},
        {"role": "user", "content": task},
    ]

    for step in range(1, 9):
        request = {"model": "gpt-4o", "messages": messages, "tools": TOOLS}
        host = OpenAIChatHost(request, lambda r: provider.chat.completions.create(**r))

        try:
            response = governor.step(host)
        except JevError as error:
            # A JevError means nothing ran, so stopping here loses no work.
            print(f"\nhalted: {error}")
            return 1

        event = governor.last
        print(f"[{step}] {event.path:16} {event.action!s:22} {event.decision_ms:6.0f} ms")

        message = response.choices[0].message
        calls = message.tool_calls or []
        if not calls:
            print(f"\n{message.content}")
            return 0

        messages.append(message.model_dump(exclude_none=True))
        for call in calls:
            arguments = json.loads(call.function.arguments or "{}")
            result = run_shell(**arguments) if call.function.name == "run_shell" else "unknown tool"
            print(f"     -> {call.function.name}({arguments}) => {result.strip()[:90]}")
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": result[:4000]}
            )

    print("\nstep limit reached")
    return 1


if __name__ == "__main__":
    # `shell=True` above is fine for a local example and wrong for anything real. In a
    # deployment the executor is the host framework's sandbox, and the risk gate is a second
    # line of defence rather than the only one.
    raise SystemExit(main())
