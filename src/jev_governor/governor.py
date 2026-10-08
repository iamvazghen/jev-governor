"""The decision loop: one System One call, then a bounded set of things that may happen.

This is the whole product. Read :func:`Governor.step` top to bottom and the architecture is
visible in the order of the guards -- the sequence is not arbitrary, and each one is placed
ahead of the next because it protects against a failure the later guards would miss.

The order, and the reason for it:

1. **Interrupted?** A decision takes a few hundred milliseconds. If the user said stop during
   it, nothing should happen, and checking before any effect is free.
2. **Truncated view plus a ``finish``?** Delegate. The one judgment that must never be made on
   a partial transcript is "the work is already done" -- the missing part is exactly where the
   unfinished work would be.
3. **Confidently high risk?** Stop and ask the human. Placed *before* the confidence gate
   deliberately: a risky action the model is unsure about must not be handed to the generative
   model as a fallback, because the fallback has fewer brakes than the thing being overridden.
4. **Unconfident, or explicitly delegated?** Hand the step to the generative model. This is the
   designed-for path, not a failure -- cheap, frequent, and the reason the fast path can afford
   to be strict.
5. **Clarify?** Ask the user, with no generative call.
6. **Finish?** Only if the independent completion judgment also clears its own, higher bar.
7. **A tool?** Either a pre-approved static payload, or the generative model is constrained to
   that one function and *checked*.

Step 7 is where this stops being advice. The host constrains the provider to the chosen tool,
and :meth:`Governor._checked` then compares what came back against what was asked for. If a
provider ignores ``tool_choice`` -- and they do, under load, with older models, and through
proxies -- the turn raises instead of executing a tool nobody selected.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .client import JevClient
from .config import GovernorConfig
from .decision import Decision
from .errors import JevError
from .host import Host
from .questions import build as build_questions
from .state import project
from .static import validate_static_arguments

#: Why a turn took the path it did. Stable strings -- safe to count in a dashboard.
PATHS = (
    "static_tool",       # routed and executed with a pre-approved payload, no generative call
    "tool_arguments",    # routed by System One, arguments written by the generative model
    "final_response",    # System One agreed the work is done; the model wrote prose only
    "clarification",     # the user was asked for missing information, no generative call
    "risk_stop",         # confidently high risk, so a human was asked first
    "delegated",         # handed to the generative model by choice or by low confidence
    "completion_held",   # a `finish` was refused because completion was not certain enough
    "truncated_hold",    # a `finish` was refused because the view was partial
    "error",             # System One failed; `on_error` decided what happened next
)

#: Keeping every event forever is a slow leak in a long-lived agent.
MAX_EVENTS = 200

_FINAL_DIRECTIVE = (
    "The decision layer has determined the substantive work is complete. Write the final "
    "user-facing answer from the results already observed. Do not call any tool."
)


def _tool_directive(name: str) -> str:
    return (
        f"The decision layer selected the function {name!r} for this step. Generate its "
        "arguments and call exactly that function. Do not select a different tool, and do not "
        "call more than one."
    )


@dataclass
class Event:
    """One recorded decision, for metrics and after-the-fact argument."""

    path: str
    decision_ms: float
    action: str | None = None
    confidence: float | None = None
    risk: str | None = None
    goal_probability: float | None = None
    model: str = ""
    usage: dict = field(default_factory=dict)
    truncated: bool = False


class Governor:
    """Routes one agent step with System One, falling back to the generative model."""

    def __init__(self, client: JevClient, config: GovernorConfig | None = None) -> None:
        self.client = client
        self.config = config or GovernorConfig()
        self.events: list[Event] = []

    # -- recording -------------------------------------------------------------------------

    def _record(
        self,
        started: float,
        path: str,
        decision: Decision | None = None,
        *,
        truncated: bool = False,
    ) -> None:
        self.events.append(
            Event(
                path=path,
                decision_ms=round((time.perf_counter() - started) * 1000, 3),
                action=decision.action if decision else None,
                confidence=decision.confidence if decision else None,
                risk=decision.risk if decision else None,
                goal_probability=decision.goal_probability if decision else None,
                model=decision.model if decision else self.config.model,
                usage=dict(decision.usage) if decision else {},
                truncated=truncated,
            )
        )
        del self.events[:-MAX_EVENTS]

    @property
    def last(self) -> Event | None:
        """The most recent event, or ``None`` before the first step."""
        return self.events[-1] if self.events else None

    # -- the decision ----------------------------------------------------------------------

    def decide(self, host: Host) -> tuple[Decision, dict]:
        """Ask System One about the host's current state. Returns the decision and projection."""
        tools = host.tools()
        questions = build_questions(tools, goal_hint=self.config.goal_hint)
        state = project(host.messages(), self.config.state_chars)
        raw = self.client.evaluate(state, questions)
        decision = Decision.parse(raw, set(questions["action"]["criteria"]))
        return decision, state

    def step(self, host: Host) -> Any:
        """Route one agent step and return the host's native response."""
        started = time.perf_counter()
        try:
            decision, state = self.decide(host)
        except JevError:
            self._record(started, "error")
            if self.config.on_error == "delegate":
                return self._delegate(host)
            raise

        if host.interrupted():
            raise InterruptedError("Interrupted after the decision and before any effect")

        return self._dispatch(host, started, decision, truncated=bool(state.get("truncated")))

    def _dispatch(self, host: Host, started: float, decision: Decision, *, truncated: bool) -> Any:
        floor = self.config.confidence_threshold

        # A partial view cannot justify stopping: the dropped history is where the
        # unfinished work would be.
        if truncated and decision.action == "finish":
            self._record(started, "truncated_hold", decision, truncated=True)
            return self._delegate(host)

        # Ahead of the confidence gate on purpose: a risky step must not fall through to the
        # generative model, which has fewer brakes than the decision being overridden.
        if decision.risk == "high" and decision.risk_confidence >= floor:
            self._record(started, "risk_stop", decision, truncated=truncated)
            return host.say(
                "The next step looks potentially destructive, irreversible, or outside clear "
                "authorization, so I have stopped before taking it. Tell me the intended scope "
                "and I will continue."
            )

        unconfident = decision.confidence < floor or decision.risk_confidence < floor
        if decision.action == "delegate" or unconfident:
            self._record(started, "delegated", decision, truncated=truncated)
            return self._delegate(host)

        if decision.action == "clarify":
            self._record(started, "clarification", decision, truncated=truncated)
            return host.say(
                "I need one more detail to continue reliably. Please tell me the target, the "
                "outcome you want, or the input that is missing."
            )

        if decision.action == "finish":
            # Asked independently of `action`, so the two can disagree -- and when they do,
            # the stricter completion judgment wins.
            if decision.goal_probability < self.config.completion_threshold:
                self._record(started, "completion_held", decision, truncated=truncated)
                return self._delegate(host)
            self._record(started, "final_response", decision, truncated=truncated)
            return self._checked(host, force_tool=None, directive=_FINAL_DIRECTIVE)

        name = decision.tool
        if name is None:
            raise JevError("System One returned an action that is neither a control nor a tool")

        arguments = self.config.static_tools.get(name)
        if arguments is not None and decision.risk == "low":
            validate_static_arguments(host.tools(), name, arguments)
            self._record(started, "static_tool", decision, truncated=truncated)
            return host.say_tool_call(name, dict(arguments))

        self._record(started, "tool_arguments", decision, truncated=truncated)
        return self._checked(host, force_tool=name, directive=_tool_directive(name))

    # -- generation -------------------------------------------------------------------------

    def _delegate(self, host: Host) -> Any:
        """Hand the step to the generative model with no constraint, as it was before."""
        return host.generate(force_tool=None, directive="")

    def _checked(self, host: Host, *, force_tool: str | None, directive: str) -> Any:
        """Generate under a constraint, then verify the provider actually honoured it."""
        response = host.generate(force_tool=force_tool, directive=directive)
        called = host.called_tools(response)

        if force_tool is None:
            if called:
                raise JevError(
                    "The provider called a tool after a no-tool decision; execution blocked"
                )
            return response
        if called != [force_tool]:
            raise JevError("The provider ignored the selected tool; execution blocked")
        return response
