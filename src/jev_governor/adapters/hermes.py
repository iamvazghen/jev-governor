"""Adapter for Nous Research's Hermes Agent.

Hermes already has the two things this package needs: a single place where every provider call
funnels through (``agent/turn_api_call.py``), and OpenAI-compatible request dicts. So the
install is a seam, not a rewrite -- six lines in one upstream file, which is small enough that
merging new Hermes releases stays mechanical. See ``docs/install-hermes.md``.

What this adapter adds over :class:`~jev_governor.adapters.openai_chat.OpenAIChatHost`:

* **Interrupt awareness.** Hermes tracks a pending interrupt and a pending redirect, and a
  decision takes long enough for either to arrive. Both are honoured before any effect.
* **Response normalisation.** Hermes transports normalise provider quirks before the agent
  reads a response; using the agent's own normaliser means ``called_tools`` sees what the agent
  will see, not a second interpretation of the same bytes.
* **An explicit API-mode check.** Hermes can run ``codex_responses``, which is a different
  request shape with different tool semantics. Rather than silently mis-parse it, this refuses.

Everything else in Hermes -- tools, sandboxes, session persistence, memory, skills, compression
-- is untouched upstream code and keeps working exactly as it did.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..errors import JevError
from .openai_chat import OpenAIChatHost

#: The only Hermes API mode whose request shape this adapter understands.
SUPPORTED_API_MODE = "chat_completions"


class HermesHost(OpenAIChatHost):
    """Governs one Hermes provider call."""

    def __init__(self, agent: Any, request: dict, generate: Callable[[dict], Any]) -> None:
        mode = getattr(agent, "api_mode", SUPPORTED_API_MODE)
        if mode != SUPPORTED_API_MODE:
            raise JevError(
                f"The Hermes adapter supports api_mode={SUPPORTED_API_MODE!r}, not {mode!r}"
            )
        super().__init__(request, generate)
        self.agent = agent

    def interrupted(self) -> bool:
        if getattr(self.agent, "_interrupt_requested", False):
            return True
        pending = getattr(self.agent, "_has_pending_redirect", None)
        return bool(pending()) if callable(pending) else False

    def called_tools(self, response: Any) -> list[str]:
        """Read tool calls through the agent's own transport normaliser where available."""
        transport = getattr(self.agent, "_get_transport", None)
        if callable(transport):
            normalise = getattr(transport(), "normalize_response", None)
            if callable(normalise):
                message = normalise(response)
                calls = getattr(message, "tool_calls", None) or []
                return [
                    call.function.name
                    for call in calls
                    if getattr(getattr(call, "function", None), "name", None)
                ]
        return super().called_tools(response)


def install(agent: Any, config: Any) -> None:
    """Attach a governor to ``agent`` from a Hermes profile config, or clear it.

    Reads a ``jev`` section from the agent config. Absent or ``enabled: false`` clears the
    governor, so the feature is opt-in per profile and switching it off is one line rather than
    an uninstall.

    .. code-block:: yaml

       jev:
         enabled: true
         confidence_threshold: 0.8
         on_error: delegate

    The credential is read from the agent's secret scope when Hermes provides one, so the key
    stays wherever Hermes already keeps secrets rather than acquiring a second home in this
    package's own settings.
    """
    from ..client import JevClient
    from ..config import GovernorConfig
    from ..governor import Governor

    values = (config or {}).get("jev", {}) if isinstance(config, dict) else {}
    if not isinstance(values, dict):
        raise JevError("The jev configuration section must be an object")
    if not values.get("enabled", False):
        agent._jev_governor = None
        return

    settings = GovernorConfig.from_dict(values)
    key = _secret(agent, "TYPESAFE_API_KEY")
    governor = Governor(
        JevClient(key or "", model=settings.model, timeout=settings.timeout), settings
    )

    def call(agent_: Any, request: dict, generate: Callable[[dict], Any]) -> Any:
        return governor.step(HermesHost(agent_, request, generate))

    governor.call = call  # type: ignore[attr-defined]
    agent._jev_governor = governor


def _secret(agent: Any, name: str) -> str:
    """Read ``name`` from Hermes's secret scope, falling back to the process environment."""
    try:
        from agent.secret_scope import current_secret_scope, get_secret  # type: ignore
    except ImportError:
        import os

        return os.environ.get(name, "")
    scope = current_secret_scope()
    if scope is not None:
        return scope.get(name, "")
    return get_secret(name) or ""
