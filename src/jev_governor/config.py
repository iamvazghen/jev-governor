"""Configuration, validated at construction rather than at the moment it matters.

Every field here is a number someone will eventually want to tune in a YAML file, which means
every field is a number someone can eventually set to nonsense. Validation happens once, when
the governor is built, so a bad threshold fails at startup with a message naming the field --
not three hours later, mid-turn, as a confident route past a gate that was never on.

The two thresholds answer different questions and are deliberately not one knob:

``confidence_threshold``
    How concentrated must the *routing* distribution be before the fast path is allowed? Below
    it, the step goes to the generative model instead. This is the dial that trades cost for
    caution.

``completion_threshold``
    How certain must the agent be that the *work is actually done* before it is allowed to stop?
    Set higher than the routing threshold by default, because the failure modes are not
    symmetric: a needless delegation costs a few cents, while a turn that stops early hands the
    user a confident answer about work that never happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .client import DEFAULT_MODEL
from .decision import probability
from .errors import JevError

#: Below this many characters the state projection cannot carry a useful turn; above it, the
#: cost of re-asking on every step stops being negligible.
MIN_STATE_CHARS = 4_000
MAX_STATE_CHARS = 100_000

#: What to do when System One itself is unreachable or incoherent.
#:
#: ``"delegate"`` -- fall back to the generative model and keep serving. The agent behaves
#: exactly as it did before this package was installed, which is the right default for an
#: assistant: a degraded answer beats no answer.
#:
#: ``"stop"`` -- raise. Correct when the governor is load-bearing for safety rather than for
#: speed, because a silent fallback would quietly remove the risk gate along with the routing.
ON_ERROR_POLICIES = ("delegate", "stop")


@dataclass(frozen=True)
class GovernorConfig:
    """Validated settings for one :class:`~jev_governor.governor.Governor`."""

    confidence_threshold: float = 0.80
    completion_threshold: float = 0.95
    state_chars: int = 24_000
    timeout: float = 5.0
    model: str = DEFAULT_MODEL
    on_error: str = "delegate"
    goal_hint: str = ""
    static_tools: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        probability(self.confidence_threshold)
        probability(self.completion_threshold)

        in_range = MIN_STATE_CHARS <= self.state_chars <= MAX_STATE_CHARS
        if type(self.state_chars) is not int or not in_range:
            raise JevError(
                f"state_chars must be an integer between {MIN_STATE_CHARS} and {MAX_STATE_CHARS}"
            )
        if isinstance(self.timeout, bool) or not isinstance(self.timeout, (int, float)):
            raise JevError("timeout must be a number of seconds")
        if not 0 < self.timeout <= 30:
            raise JevError("timeout must be greater than zero and at most 30 seconds")
        if not isinstance(self.model, str) or not self.model:
            raise JevError("model must be a non-empty model id")
        if self.on_error not in ON_ERROR_POLICIES:
            raise JevError(f"on_error must be one of {', '.join(ON_ERROR_POLICIES)}")
        if not isinstance(self.goal_hint, str):
            raise JevError("goal_hint must be a string")

        if not isinstance(self.static_tools, dict):
            raise JevError("static_tools must map tool names to argument objects")
        for name, arguments in self.static_tools.items():
            if not isinstance(name, str) or not name:
                raise JevError("static_tools keys must be tool names")
            if not isinstance(arguments, dict):
                raise JevError(f"static_tools[{name!r}] must be an argument object")

    @classmethod
    def from_dict(cls, values: Any) -> GovernorConfig:
        """Build from a plain mapping, rejecting unknown keys.

        Unknown keys are an error rather than a warning on purpose. A typo in
        ``confidence_threshold`` that is silently ignored leaves the gate at its default while
        the operator believes it was tightened, and nothing in the logs says otherwise.
        """
        if not isinstance(values, dict):
            raise JevError("Configuration must be an object")
        known = set(cls.__dataclass_fields__)
        # `enabled` is consumed by host wiring, not by the governor itself.
        unknown = set(values) - known - {"enabled"}
        if unknown:
            raise JevError(f"Unknown configuration fields: {', '.join(sorted(unknown))}")
        return cls(**{key: value for key, value in values.items() if key in known})
