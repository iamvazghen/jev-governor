"""The one error type, and what it guarantees.

``JevError`` has a single meaning, and the whole package depends on it: **no decision was
executed.** It is raised before any effect, never after one. That is what lets a caller treat
it as safe to retry, and what lets :class:`~jev_governor.config.GovernorConfig`'s ``on_error``
policy be a two-line choice rather than a taxonomy.

Messages never carry a remote response body, a credential, or a fragment of agent state. A
decision layer that leaks the thing it was shown is worse than no decision layer, and an
exception string is the easiest place in a Python program for that to happen by accident.
"""

from __future__ import annotations


class JevError(RuntimeError):
    """A decision could not be trusted, so nothing was run.

    Never include remote bodies, credentials, or transcript content in the message.
    """
