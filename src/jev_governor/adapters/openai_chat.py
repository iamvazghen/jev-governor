"""A :class:`~jev_governor.host.Host` for any OpenAI-compatible chat-completions agent.

This is the adapter most people want, and the one the framework-specific adapters are built on.
If your agent loop ends in "assemble a request dict, POST it, read ``tool_calls``", this governs
it without touching the loop.

The one thing to understand before using it: **``force_tool`` has to reach the provider as a
real constraint, not a suggestion in the prompt.** This adapter sets
``tool_choice={"type": "function", "function": {"name": ...}}`` and
``parallel_tool_calls=False``, which every OpenAI-compatible provider worth using honours --
and then :class:`~jev_governor.governor.Governor` verifies the result anyway, because some
providers quietly do not.

Two properties that keep the cost where it should be:

* The directive is appended **to a copy** of the request. The canonical message list the caller
  owns is never mutated, so its cacheable prefix stays byte-identical and the prompt cache
  keeps working. A decision layer that invalidates the cache every turn costs more than it
  saves.
* ``say`` and ``say_tool_call`` build a response locally with no network call at all. Phrasing
  a refusal or firing a pre-approved payload should not cost a generative round-trip.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from typing import Any

from ..errors import JevError

#: Model id stamped on locally-built responses, so they are identifiable in logs and are
#: never mistaken for provider output.
SYNTHETIC_MODEL = "jev-governor-local"


def _as_openai_object(payload: dict) -> Any:
    """Return ``payload`` as an ``openai`` ChatCompletion when that package is installed.

    Agent frameworks tend to call attributes on the response (``.choices[0].message``), so a
    bare dict would break them. When ``openai`` is absent -- a provider-agnostic host, a test
    -- the dict is returned as-is and documented as such.
    """
    try:
        from openai.types.chat import ChatCompletion
    except ImportError:
        return payload
    return ChatCompletion.model_validate(payload)


def _completion(*, content: str | None, tool_calls: list[dict] | None) -> Any:
    return _as_openai_object(
        {
            "id": "jevgov_" + uuid.uuid4().hex,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": SYNTHETIC_MODEL,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls,
                    },
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    )


class OpenAIChatHost:
    """Governs one chat-completions request.

    ``request`` is the request body you were about to send. ``generate`` is whatever actually
    sends it -- an ``openai`` client call, an ``httpx`` post, your framework's transport. It
    receives a (possibly narrowed) copy of the request and returns the provider response
    unchanged.
    """

    def __init__(
        self,
        request: dict,
        generate: Callable[[dict], Any],
        *,
        interrupted: Callable[[], bool] | None = None,
    ) -> None:
        if not isinstance(request, dict):
            raise JevError("request must be a chat-completions request object")
        self.request = request
        self._generate = generate
        self._interrupted = interrupted

    # -- reading ---------------------------------------------------------------------------

    def tools(self) -> list[dict]:
        tools = self.request.get("tools") or []
        if not isinstance(tools, list):
            raise JevError("request['tools'] must be a list of tool schemas")
        return tools

    def messages(self) -> list[dict]:
        messages = self.request.get("messages") or []
        if not isinstance(messages, list):
            raise JevError("request['messages'] must be a list of messages")
        return messages

    def interrupted(self) -> bool:
        return bool(self._interrupted()) if self._interrupted is not None else False

    # -- generating ------------------------------------------------------------------------

    def generate(self, *, force_tool: str | None, directive: str) -> Any:
        narrowed = dict(self.request)

        if directive:
            # Appended to a copy. The caller's list keeps its identity and its cache prefix.
            narrowed["messages"] = [*self.messages(), {"role": "user", "content": directive}]

        if force_tool is not None:
            narrowed["tool_choice"] = {"type": "function", "function": {"name": force_tool}}
            narrowed["parallel_tool_calls"] = False
        elif directive:
            # A directive with no tool means the decision was `finish`: prose only.
            narrowed["tool_choice"] = "none"

        return self._generate(narrowed)

    # -- inspecting ------------------------------------------------------------------------

    def called_tools(self, response: Any) -> list[str]:
        """Return the called function names, accepting both object and dict responses."""
        choices = _get(response, "choices") or []
        if not choices:
            return []
        message = _get(choices[0], "message")
        calls = _get(message, "tool_calls") or []
        names = []
        for call in calls:
            function = _get(call, "function")
            name = _get(function, "name")
            if isinstance(name, str):
                names.append(name)
        return names

    # -- answering without the model -------------------------------------------------------

    def say(self, text: str) -> Any:
        return _completion(content=text, tool_calls=None)

    def say_tool_call(self, name: str, arguments: dict) -> Any:
        return _completion(
            content=None,
            tool_calls=[
                {
                    "id": "call_jevgov_" + uuid.uuid4().hex,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ],
        )


def _get(obj: Any, key: str) -> Any:
    """Read ``key`` from a dict or an attribute from an object. Providers return both."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)
