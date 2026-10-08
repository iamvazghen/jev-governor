"""jev-governor -- put a calibrated System One decision layer on top of an existing agent.

The idea in four lines. A normal agent loop asks a generative model to do two different jobs at
once: decide what to do next, and write the thing. The first job is a classification over a
fixed set of options, which is the one thing a non-autoregressive judgment model does in a few
hundred milliseconds for a fraction of a cent. So the loop splits -- System One decides, the
generative model only writes -- and the generative model is then *constrained* to the decision
and checked against it.

What that buys, and what it does not:

* It is faster and cheaper on the routing step, because no tokens are generated to pick an
  option and no tool descriptions are re-sent to choose between them.
* It makes the control flow inspectable. Every turn records which of nine paths it took, with
  the probabilities behind it, so "why did it do that" has an answer that is not a transcript.
* It adds a risk gate and a completion gate that the generative path did not have.
* It does **not** make judgments correct, and confidence is not permission. Everything here is
  about making an incoherent or low-confidence decision *loud* rather than silent.

Getting started is one of three lines, depending on your framework::

    from jev_governor.adapters.pydantic_ai import GovernedModel   # wrap a model
    from jev_governor.adapters.hermes import install              # patch one seam
    from jev_governor.adapters.openclaw import OpenClawRouter     # route a fleet

Or govern any OpenAI-compatible loop directly::

    from jev_governor import Governor, JevClient
    from jev_governor.adapters.openai_chat import OpenAIChatHost

    governor = Governor(JevClient(key))
    response = governor.step(OpenAIChatHost(request, send_it))

See ``docs/architecture.md`` for the reasoning and ``docs/limits.md`` for the honest list of
what this cannot do.
"""

from __future__ import annotations

from .client import DEFAULT_MODEL, AsyncJevClient, JevClient
from .config import ON_ERROR_POLICIES, GovernorConfig
from .decision import RISK_LEVELS, Decision, choice, noul, probability
from .errors import JevError
from .governor import MAX_EVENTS, PATHS, Event, Governor
from .host import Host
from .questions import CONTROL_OPTIONS, MAX_CRITERIA
from .questions import build as build_questions
from .state import project

__version__ = "0.1.0"

__all__ = [
    "CONTROL_OPTIONS",
    "DEFAULT_MODEL",
    "MAX_CRITERIA",
    "MAX_EVENTS",
    "ON_ERROR_POLICIES",
    "PATHS",
    "RISK_LEVELS",
    "AsyncJevClient",
    "Decision",
    "Event",
    "Governor",
    "GovernorConfig",
    "Host",
    "JevClient",
    "JevError",
    "__version__",
    "build_questions",
    "choice",
    "noul",
    "probability",
    "project",
]
