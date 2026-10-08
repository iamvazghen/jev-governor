"""Parsing and distrusting a System One answer.

Everything here exists because a typed answer is not the same as a *trustworthy* answer. The
service returns JSON that matches its schema; this module decides whether that JSON is
self-consistent enough to route on. Four properties are enforced, and each one has a concrete
failure behind it:

1. A probability is a finite real in ``[0, 1]`` and is not a ``bool``. In Python
   ``isinstance(True, int)`` is true, so a stray boolean would otherwise pass every numeric
   check below and compare equal to full certainty.
2. A choice distribution covers exactly the offered options -- no extras, none missing -- and
   sums to one. A distribution over a *different* option set means the question and the answer
   have drifted apart, which is precisely when routing must stop.
3. The returned ``choice`` is the distribution's argmax. If the two disagree, the answer is
   internally inconsistent and neither half can be believed.
4. Reported confidence is clamped to the selected option's own probability. Confidence
   summarises how concentrated the distribution is, so it can only ever be more optimistic
   than the mass actually sitting on the winner. The minimum is the honest number.

None of this makes a judgment correct. It makes an *incoherent* judgment loud instead of
silent, which is the only guarantee a client library can honestly offer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .errors import JevError

#: Risk is triage, not authorization. See :mod:`jev_governor.questions`.
RISK_LEVELS = ("low", "medium", "high")


def probability(value: Any) -> float:
    """Return ``value`` as a probability, or raise.

    ``bool`` is rejected explicitly: it is a subclass of ``int``, so ``True`` would otherwise
    pass every numeric guard below and be read as certainty.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JevError("System One returned a non-numeric probability")
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise JevError("System One returned an out-of-range probability")
    return float(value)


def choice(answer: Any, options: set[str]) -> tuple[str, float]:
    """Return ``(selected, confidence)`` from a ``choice`` answer over exactly ``options``."""
    if not isinstance(answer, dict):
        raise JevError("System One returned a non-object answer")
    if answer.get("type") != "choice":
        raise JevError("System One returned the wrong primitive for a choice question")
    selected = answer.get("choice")
    if selected not in options:
        raise JevError("System One chose an option that was not offered")

    distribution = answer.get("probabilities")
    if not isinstance(distribution, dict) or set(distribution) != options:
        raise JevError("System One returned a distribution over a different option set")
    mass = {key: probability(value) for key, value in distribution.items()}
    if not math.isclose(sum(mass.values()), 1.0, abs_tol=0.01):
        raise JevError("System One probabilities do not sum to one")
    # A small epsilon keeps floating-point ties from reading as real disagreement.
    if mass[selected] + 1e-6 < max(mass.values()):
        raise JevError("System One's choice disagrees with its own distribution")

    stated = probability(answer.get("confidence"))
    return selected, min(stated, mass[selected])


def noul(answer: Any) -> float:
    """Return the probability from a ``noul`` answer.

    A ``noul`` carries no ``confidence`` field -- the probability *is* the answer. A value near
    0.5 means "about as likely as not", which is not the same as "moderately yes". Callers that
    read it as an intensity will misinterpret it.
    """
    if not isinstance(answer, dict):
        raise JevError("System One returned a non-object answer")
    if answer.get("type") != "noul":
        raise JevError("System One returned the wrong primitive for a noul question")
    return probability(answer.get("noul"))


@dataclass(frozen=True)
class Decision:
    """One routing verdict: what to do next, how sure, and how risky.

    ``confidence`` and ``risk_confidence`` describe distribution concentration. Neither is the
    probability that the action is *correct*, and neither is permission to act.
    """

    action: str
    confidence: float
    goal_probability: float
    risk: str
    risk_confidence: float
    model: str
    usage: dict

    @property
    def tool(self) -> str | None:
        """The selected tool name, or ``None`` when the action is not a tool call."""
        return self.action.removeprefix("tool:") if self.action.startswith("tool:") else None

    @classmethod
    def parse(cls, response: Any, options: set[str]) -> Decision:
        """Parse a full System One response body into a :class:`Decision`."""
        if not isinstance(response, dict):
            raise JevError("System One returned a non-object response")
        try:
            answers = response["answers"]
            action, confidence = choice(answers["action"], options)
            risk, risk_confidence = choice(answers["risk"], set(RISK_LEVELS))
            goal_probability = noul(answers["goal_met"])

            model = response["model"]
            usage = response["usage"]
            if not isinstance(model, str) or not model:
                raise JevError("System One response is missing its resolved model id")
            if not isinstance(usage, dict):
                raise JevError("System One response is missing usage metadata")
            for key in ("input_tokens", "output_tokens"):
                # `type(...) is not int` rather than isinstance: a bool token count is corrupt.
                if type(usage.get(key)) is not int or usage[key] < 0:
                    raise JevError("System One returned invalid usage metadata")
        except (KeyError, TypeError, AttributeError) as exc:
            raise JevError("Malformed System One response") from exc

        return cls(
            action=action,
            confidence=confidence,
            goal_probability=goal_probability,
            risk=risk,
            risk_confidence=risk_confidence,
            model=model,
            usage=dict(usage),
        )
