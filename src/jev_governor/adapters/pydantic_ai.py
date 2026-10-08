"""Adapter for Pydantic AI, as a model wrapper.

Pydantic AI gives this package the cleanest install of the three: a documented
``WrapperModel`` base whose ``request`` sits between the agent graph and the real provider.
Wrapping a model means **no change to your agent, your tools, or your run calls** -- you build
the agent with ``GovernedModel('openai:gpt-4o')`` instead of ``'openai:gpt-4o'`` and nothing
else moves.

How the constraint is applied, which is the part specific to this framework. Pydantic AI has no
``tool_choice`` field; it decides what the model may do from the request parameters. So:

* To force one tool, the request is re-issued with ``function_tools`` filtered to just that
  tool and ``allow_text_output=False``. The model has exactly one legal move.
* To finish, the request is re-issued with ``function_tools=[]`` and ``allow_text_output=True``.
  Prose is the only legal move.

That is strictly stronger than asking politely in the prompt, and it uses only documented
fields rather than reaching into provider internals. ``Governor`` still verifies the result.

One honest limitation: output tools (``output_type=`` structured results) are left untouched in
both cases. They are how Pydantic AI returns a typed result rather than a side effect, so
removing them would break the agent's contract with its caller, and governing them would mean
deciding whether the run may return at all -- a different question from the one this package
asks.
"""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any

from ..client import AsyncJevClient, JevClient
from ..config import GovernorConfig
from ..errors import JevError
from ..governor import Governor

try:  # pragma: no cover - import shape depends on the installed extra
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
    from pydantic_ai.models.wrapper import WrapperModel

    _BASE: Any = WrapperModel
    _AVAILABLE = True
except ImportError:  # pragma: no cover
    _BASE = object
    _AVAILABLE = False


def _require_pydantic_ai() -> None:
    if not _AVAILABLE:
        raise JevError(
            "The Pydantic AI adapter needs pydantic-ai "
            "(pip install 'jev-governor[pydantic-ai]')"
        )


def _tool_schemas(parameters: Any) -> list[dict]:
    """Project Pydantic AI ``ToolDefinition`` objects into OpenAI-style tool schemas."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.parameters_json_schema,
            },
        }
        for tool in getattr(parameters, "function_tools", []) or []
    ]


def _messages(messages: list[Any]) -> list[dict]:
    """Project Pydantic AI messages into the OpenAI-ish rows the state projection reads.

    Only text and tool calls are carried. A ``ModelRequest`` holds the user and system turns,
    a ``ModelResponse`` the assistant turn; both are flattened part by part.
    """
    rows: list[dict] = []
    for message in messages:
        for part in getattr(message, "parts", []) or []:
            kind = getattr(part, "part_kind", "")
            if kind == "system-prompt":
                rows.append({"role": "system", "content": getattr(part, "content", "")})
            elif kind == "user-prompt":
                content = getattr(part, "content", "")
                rows.append(
                    {"role": "user", "content": content if isinstance(content, str) else ""}
                )
            elif kind == "text":
                rows.append({"role": "assistant", "content": getattr(part, "content", "")})
            elif kind == "tool-call":
                rows.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": getattr(part, "tool_name", ""),
                                    "arguments": str(getattr(part, "args", "")),
                                },
                            }
                        ],
                    }
                )
            elif kind == "tool-return":
                rows.append(
                    {
                        "role": "tool",
                        "name": getattr(part, "tool_name", ""),
                        "content": str(getattr(part, "content", "")),
                    }
                )
    return rows


class _Host:
    """A :class:`~jev_governor.host.Host` over one Pydantic AI request.

    ``generate`` must be synchronous for the governor, but the wrapped model is async, so the
    coroutine is run on a private loop in a worker thread. The governor is called from
    ``asyncio.to_thread`` in :meth:`GovernedModel.request`, so the caller's loop is never
    blocked and this never re-enters it.
    """

    def __init__(self, model: Any, messages: list[Any], settings: Any, parameters: Any) -> None:
        self.model = model
        self._messages = messages
        self._settings = settings
        self._parameters = parameters
        self.response: Any = None

    def tools(self) -> list[dict]:
        return _tool_schemas(self._parameters)

    def messages(self) -> list[dict]:
        return _messages(self._messages)

    def interrupted(self) -> bool:
        return False  # Pydantic AI cancels the task itself; there is no flag to consult.

    def generate(self, *, force_tool: str | None, directive: str) -> Any:
        parameters = self._parameters
        messages = self._messages

        if directive:
            messages = [*messages, _user_request(directive)]

        if force_tool is not None:
            kept = [
                tool
                for tool in getattr(parameters, "function_tools", []) or []
                if tool.name == force_tool
            ]
            parameters = dataclasses.replace(
                parameters, function_tools=kept, allow_text_output=False
            )
        elif directive:
            parameters = dataclasses.replace(
                parameters, function_tools=[], allow_text_output=True
            )

        coroutine = self.model.wrapped.request(messages, self._settings, parameters)
        return asyncio.run(coroutine)

    def called_tools(self, response: Any) -> list[str]:
        return [
            part.tool_name
            for part in getattr(response, "parts", []) or []
            if isinstance(part, ToolCallPart)
        ]

    def say(self, text: str) -> Any:
        return ModelResponse(parts=[TextPart(content=text)], model_name="jev-governor-local")

    def say_tool_call(self, name: str, arguments: dict) -> Any:
        return ModelResponse(
            parts=[ToolCallPart(tool_name=name, args=arguments)],
            model_name="jev-governor-local",
        )


def _user_request(text: str) -> Any:
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    return ModelRequest(parts=[UserPromptPart(content=text)])


class GovernedModel(_BASE):  # type: ignore[misc,valid-type]
    """A Pydantic AI model that routes each step through System One first.

    .. code-block:: python

       from pydantic_ai import Agent
       from jev_governor.adapters.pydantic_ai import GovernedModel

       agent = Agent(GovernedModel('openai:gpt-4o', api_key=key), tools=[...])
       result = agent.run_sync('Check whether port 8080 is listening')

    ``api_key`` defaults to ``TYPESAFE_API_KEY`` from the environment, so a deployment that
    already sets it needs no argument at all.
    """

    def __init__(
        self,
        wrapped: Any,
        *,
        api_key: str | None = None,
        config: GovernorConfig | None = None,
        client: JevClient | AsyncJevClient | None = None,
    ) -> None:
        _require_pydantic_ai()
        super().__init__(wrapped)
        settings = config or GovernorConfig()
        if client is None:
            import os

            key = api_key if api_key is not None else os.environ.get("TYPESAFE_API_KEY", "")
            client = JevClient(key, model=settings.model, timeout=settings.timeout)
        self.governor = Governor(client, settings)

    async def request(
        self, messages: list[Any], model_settings: Any, model_request_parameters: Any
    ) -> Any:
        host = _Host(self, messages, model_settings, model_request_parameters)
        # The governor and the wrapped model are both synchronous from here, so the whole
        # step runs off the event loop rather than blocking it.
        return await asyncio.to_thread(self.governor.step, host)
