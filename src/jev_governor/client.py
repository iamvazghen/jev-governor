"""A small, deliberately boring HTTP client for the System One endpoint.

Why hand-rolled rather than depending on the official ``typesafe`` SDK: the wire protocol used
here is one POST with three documented fields, and this package wants exactly one runtime
dependency so it can drop into someone else's agent without a version negotiation. If you would
rather use the official SDK -- or a shared connection pool, or a recording proxy in tests --
pass any object with a compatible ``post`` through the ``transport`` argument and this module
gets out of the way.

Three behaviours are worth knowing:

* **Retry is for congestion only.** ``429`` and ``529`` get one short retry; every other
  non-200 raises immediately. Retrying a ``422`` cannot help -- a malformed question stays
  malformed -- and retrying a ``401`` just burns the rate limit on a dead credential.
* **The key never appears in an exception.** Status codes do, response bodies do not. See
  :mod:`jev_governor.errors`.
* **Auth failures are distinguishable and that is useful.** A blank key gives ``403``
  ("must supply an API key"); a present-but-wrong key gives ``401``. So a ``401`` proves the
  header form is right and the credential itself is dead, which is a much faster diagnosis
  than a generic "auth failed".
"""

from __future__ import annotations

import time
from typing import Any

from .errors import JevError

ENDPOINT = "https://api.typesafe.ai/v1/systemone"

#: ``jev-latest`` resolves server-side; answers come back stamped with the concrete version
#: (for example ``jev-1.13.0``), which :class:`~jev_governor.decision.Decision` records so a
#: behaviour change is attributable after the fact.
DEFAULT_MODEL = "jev-latest"

#: Statuses worth one retry. Anything else is a decision the server already made.
_RETRY_STATUSES = frozenset({429, 529})
_RETRY_DELAY_S = 0.25


def _require_ok(status: int) -> None:
    if status == 403:
        raise JevError("System One rejected the request: no API key was supplied")
    if status == 401:
        raise JevError("System One rejected the API key; the header form was accepted")
    if status == 422:
        raise JevError("System One rejected the question shape (HTTP 422); no decision was made")
    if status in _RETRY_STATUSES:
        # Reached only after the retry, so this is congestion that did not clear. Saying so
        # beats a bare status code: the caller should back off, not investigate a bug.
        raise JevError(f"System One stayed congested (HTTP {status}); no decision was executed")
    if status != 200:
        raise JevError(f"System One returned HTTP {status}; no decision was executed")


def _body(payload: Any) -> dict:
    if not isinstance(payload, dict):
        raise JevError("System One returned a non-object response")
    return payload


class JevClient:
    """Synchronous System One client.

    The HTTP connection pool is created lazily and owned by this instance, never by module or
    process state, so an agent that runs several profiles with different credentials cannot
    accidentally share one.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_MODEL,
        timeout: float = 5.0,
        transport: Any = None,
        endpoint: str = ENDPOINT,
    ) -> None:
        if not api_key:
            raise JevError("No System One API key was supplied")
        self.model = model
        self.endpoint = endpoint
        self._key = api_key
        self._timeout = timeout
        self._transport = transport
        self._owned = transport is None

    def _ensure_transport(self) -> Any:
        if self._transport is None:
            import httpx

            self._transport = httpx.Client(timeout=self._timeout, follow_redirects=False)
        return self._transport

    def evaluate(self, state: Any, questions: dict) -> dict:
        """Ask ``questions`` about ``state`` and return the raw response body."""
        transport = self._ensure_transport()
        payload = {"model": self.model, "state": state, "questions": questions}
        headers = {"Authorization": f"Bearer {self._key}"}

        for attempt in (0, 1):
            try:
                response = transport.post(
                    self.endpoint, headers=headers, json=payload, timeout=self._timeout
                )
            except Exception as exc:  # transport-level: connect, read, DNS, TLS
                raise JevError("System One transport failed; no decision was executed") from exc

            status = getattr(response, "status_code", 0)
            if status in _RETRY_STATUSES and attempt == 0:
                time.sleep(_RETRY_DELAY_S)
                continue
            _require_ok(status)
            try:
                return _body(response.json())
            except JevError:
                raise
            except Exception as exc:
                raise JevError("System One returned a body that is not JSON") from exc

        raise JevError("System One stayed congested; no decision was executed")

    def close(self) -> None:
        """Release the pool if this client made it, and drop the key reference."""
        if self._owned and self._transport is not None:
            closer = getattr(self._transport, "close", None)
            if callable(closer):
                closer()
            self._transport = None
        self._key = ""

    def __enter__(self) -> JevClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class AsyncJevClient(JevClient):
    """Asynchronous System One client, same contract as :class:`JevClient`."""

    def _ensure_transport(self) -> Any:
        if self._transport is None:
            import httpx

            self._transport = httpx.AsyncClient(timeout=self._timeout, follow_redirects=False)
        return self._transport

    async def evaluate(self, state: Any, questions: dict) -> dict:  # type: ignore[override]
        import asyncio

        transport = self._ensure_transport()
        payload = {"model": self.model, "state": state, "questions": questions}
        headers = {"Authorization": f"Bearer {self._key}"}

        for attempt in (0, 1):
            try:
                response = await transport.post(
                    self.endpoint, headers=headers, json=payload, timeout=self._timeout
                )
            except Exception as exc:
                raise JevError("System One transport failed; no decision was executed") from exc

            status = getattr(response, "status_code", 0)
            if status in _RETRY_STATUSES and attempt == 0:
                await asyncio.sleep(_RETRY_DELAY_S)
                continue
            _require_ok(status)
            try:
                return _body(response.json())
            except JevError:
                raise
            except Exception as exc:
                raise JevError("System One returned a body that is not JSON") from exc

        raise JevError("System One stayed congested; no decision was executed")

    async def aclose(self) -> None:
        """Release the pool if this client made it, and drop the key reference."""
        if self._owned and self._transport is not None:
            closer = getattr(self._transport, "aclose", None)
            if callable(closer):
                await closer()
            self._transport = None
        self._key = ""

    async def __aenter__(self) -> AsyncJevClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()
