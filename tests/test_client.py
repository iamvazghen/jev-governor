"""Transport behaviour: what gets retried, what does not, and what never leaks."""

from __future__ import annotations

import pytest

from conftest import FakeResponse, FakeTransport, body
from jev_governor import AsyncJevClient, JevClient, JevError
from jev_governor.client import ENDPOINT

QUESTIONS = {"action": {"type": "choice", "instructions": "x", "criteria": {"a": "A"}}}


class TestTheRequest:
    def test_posts_the_documented_shape_to_the_documented_endpoint(self):
        transport = FakeTransport(FakeResponse(200, body()))
        JevClient("k", transport=transport).evaluate({"goal": "g"}, QUESTIONS)

        request = transport.requests[0]
        assert request["url"] == ENDPOINT
        assert request["payload"] == {
            "model": "jev-latest",
            "state": {"goal": "g"},
            "questions": QUESTIONS,
        }

    def test_carries_the_key_as_a_bearer_header(self):
        transport = FakeTransport(FakeResponse(200, body()))
        JevClient("secret-key", transport=transport).evaluate({}, QUESTIONS)
        assert transport.requests[0]["headers"] == {"Authorization": "Bearer secret-key"}

    def test_refuses_to_be_built_without_a_key(self):
        # Failing at construction beats failing mid-turn on the first real decision.
        with pytest.raises(JevError, match="API key"):
            JevClient("")


class TestRetry:
    @pytest.mark.parametrize("status", [429, 529])
    def test_retries_congestion_exactly_once(self, status):
        transport = FakeTransport(FakeResponse(status, {}), FakeResponse(200, body()))
        JevClient("k", transport=transport).evaluate({}, QUESTIONS)
        assert len(transport.requests) == 2

    def test_gives_up_when_congestion_persists(self):
        transport = FakeTransport(FakeResponse(429, {}), FakeResponse(429, {}))
        with pytest.raises(JevError, match="congested"):
            JevClient("k", transport=transport).evaluate({}, QUESTIONS)

    @pytest.mark.parametrize("status", [400, 401, 403, 422, 500, 503])
    def test_does_not_retry_anything_else(self, status):
        # Retrying a 422 cannot help: a malformed question stays malformed. Retrying a 401
        # just burns the rate limit on a dead credential.
        transport = FakeTransport(FakeResponse(status, {}))
        with pytest.raises(JevError):
            JevClient("k", transport=transport).evaluate({}, QUESTIONS)
        assert len(transport.requests) == 1


class TestDiagnosis:
    def test_a_403_says_no_key_was_supplied(self):
        transport = FakeTransport(FakeResponse(403, {}))
        with pytest.raises(JevError, match="no API key"):
            JevClient("k", transport=transport).evaluate({}, QUESTIONS)

    def test_a_401_says_the_header_form_was_accepted(self):
        # The distinction matters: a 401 proves the request shape is right and the credential
        # itself is dead, which is a much faster diagnosis than "auth failed".
        transport = FakeTransport(FakeResponse(401, {}))
        with pytest.raises(JevError, match="header form was accepted"):
            JevClient("k", transport=transport).evaluate({}, QUESTIONS)

    def test_a_422_names_the_question_shape(self):
        transport = FakeTransport(FakeResponse(422, {}))
        with pytest.raises(JevError, match="question shape"):
            JevClient("k", transport=transport).evaluate({}, QUESTIONS)


class TestLeaks:
    def test_a_transport_failure_does_not_carry_the_key(self):
        class Broken:
            def post(self, *_a, **_k):
                raise OSError("connection reset to host with Bearer super-secret")

        with pytest.raises(JevError) as caught:
            JevClient("super-secret", transport=Broken()).evaluate({}, QUESTIONS)
        assert "super-secret" not in str(caught.value)

    def test_an_http_failure_does_not_carry_the_response_body(self):
        transport = FakeTransport(FakeResponse(500, {"error": "internal detail leaked here"}))
        with pytest.raises(JevError) as caught:
            JevClient("k", transport=transport).evaluate({}, QUESTIONS)
        assert "internal detail" not in str(caught.value)

    def test_closing_drops_the_key_reference(self):
        transport = FakeTransport(FakeResponse(200, body()))
        client = JevClient("k", transport=transport)
        client.close()
        assert client._key == ""


class TestBadBodies:
    def test_rejects_a_non_json_body(self):
        class NotJson(FakeResponse):
            def json(self):
                raise ValueError("not json")

        transport = FakeTransport(NotJson(200, None))
        with pytest.raises(JevError, match="not JSON"):
            JevClient("k", transport=transport).evaluate({}, QUESTIONS)

    def test_rejects_a_json_array(self):
        transport = FakeTransport(FakeResponse(200, [1, 2, 3]))
        with pytest.raises(JevError, match="non-object"):
            JevClient("k", transport=transport).evaluate({}, QUESTIONS)


class TestOwnership:
    def test_an_injected_transport_is_not_closed_by_this_client(self):
        # The caller owns a pool it supplied; closing it would break their other requests.
        transport = FakeTransport(FakeResponse(200, body()))
        JevClient("k", transport=transport).close()
        assert transport.closed is False

    def test_the_context_manager_closes_the_client(self):
        transport = FakeTransport(FakeResponse(200, body()))
        with JevClient("k", transport=transport) as client:
            client.evaluate({}, QUESTIONS)
        assert client._key == ""


class TestAsync:
    async def test_posts_and_parses(self):
        class AsyncTransport:
            def __init__(self):
                self.calls = 0

            async def post(self, *_a, **_k):
                self.calls += 1
                return FakeResponse(200, body())

        transport = AsyncTransport()
        result = await AsyncJevClient("k", transport=transport).evaluate({}, QUESTIONS)
        assert transport.calls == 1
        assert result["model"] == "jev-1.13.0"

    async def test_retries_congestion_once(self):
        class Flaky:
            def __init__(self):
                self.statuses = [429, 200]

            async def post(self, *_a, **_k):
                status = self.statuses.pop(0)
                return FakeResponse(status, body() if status == 200 else {})

        flaky = Flaky()
        await AsyncJevClient("k", transport=flaky).evaluate({}, QUESTIONS)
        assert flaky.statuses == []
