"""The seam between this package and whatever agent it is governing.

A ``Host`` is the entire surface an agent framework has to expose. There are six methods, none
of which mention System One, so an adapter author never needs to understand the decision layer
to write one -- and this package never needs to import a framework to govern it.

Why a protocol rather than a base class to inherit: the frameworks worth supporting
(Hermes, PydanticAI, OpenClaw) already have their own class hierarchies and their own opinions
about construction. Structural typing lets an adapter be a thin wrapper around an existing
object instead of asking anyone to re-parent theirs.

The method that carries the actual architecture is :meth:`Host.generate`. Everything else is
reading. ``generate`` is where the host must *constrain* the generative model to the tool the
decision layer picked -- and then :meth:`Host.called_tools` is how the governor checks that the
constraint was honoured. Without that pair, "System One decides" is a comment rather than a
property: the provider would remain free to call something else, and the governor would be
offering advice it has no way to enforce.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Host(Protocol):
    """What an agent framework must provide to be governed."""

    def tools(self) -> list[dict]:
        """Return the tools available *for this step*, as OpenAI-style schemas.

        Called fresh on every decision, so a toolset that changes with context stays correct
        without a second registry to keep in sync.
        """
        ...

    def messages(self) -> list[dict]:
        """Return the conversation so far, oldest-first, in OpenAI message form.

        The governor only ever reads this and builds a throwaway projection from it. It will
        not be mutated.
        """
        ...

    def generate(self, *, force_tool: str | None, directive: str) -> Any:
        """Run the generative model for one step and return the framework's native response.

        ``force_tool`` is the contract. When it is a tool name, the host **must** constrain the
        provider to call exactly that function once -- with OpenAI-compatible APIs that is
        ``tool_choice={"type": "function", "function": {"name": ...}}`` plus
        ``parallel_tool_calls=False``. When it is ``None``, the host must forbid tool calls
        entirely (``tool_choice="none"``), because the decision layer has concluded the work is
        done and only prose remains.

        ``directive`` is a short instruction to append **to this request only**. Appending
        rather than rewriting keeps the canonical history and its cacheable prefix intact,
        which is the difference between a decision layer that costs nothing and one that
        invalidates the prompt cache on every turn.
        """
        ...

    def called_tools(self, response: Any) -> list[str]:
        """Return the tool names the provider actually called in ``response``.

        Used to verify the constraint in :meth:`generate` was honoured. Return an empty list
        for a plain prose reply.
        """
        ...

    def say(self, text: str) -> Any:
        """Return a native response carrying ``text`` as the assistant's reply, with no model call.

        Used for the two decisions that need no generation: a confident high-risk stop, and a
        request for clarification. Both are cases where spending a generative call to phrase a
        refusal would be the cost this package exists to avoid.
        """
        ...

    def say_tool_call(self, name: str, arguments: dict) -> Any:
        """Return a native response that calls ``name`` with ``arguments``, with no model call.

        This is the fast path: a tool whose arguments are fixed configuration rather than
        something that has to be written each time (``docker ps``, ``git status``,
        ``read_file`` on a known path). The decision cost is one System One call and the
        generative cost is zero.

        Only reached for tools listed in ``static_tools`` *and* judged low risk, and the
        payload is validated against the tool's real JSON schema first -- see
        :mod:`jev_governor.static`.
        """
        ...

    def interrupted(self) -> bool:
        """Return whether the user has asked to stop since the decision began.

        A decision takes a few hundred milliseconds, which is long enough for someone to hit
        Ctrl-C. Checking after the decision and before the effect is the cheapest honest place
        to notice.
        """
        ...
