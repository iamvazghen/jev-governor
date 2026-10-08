"""The guard order. This is the file that says what the package actually promises.

Each test names a property in its own terms rather than restating the implementation, because
the point of a guard is the behaviour it forces, not the branch it occupies.
"""

from __future__ import annotations

import pytest

from conftest import FakeHost, FakeTransport, body, spread
from jev_governor import Governor, GovernorConfig, JevClient, JevError

OPTIONS = {"finish", "clarify", "delegate", "tool:read_file"}


def governor(*bodies, **config):
    """A governor wired to a fake transport that will return ``bodies`` in order."""
    from conftest import FakeResponse

    transport = FakeTransport(*[FakeResponse(200, b) for b in bodies])
    client = JevClient("test-key", transport=transport)
    return Governor(client, GovernorConfig(**config)), transport


class TestTheFastPath:
    def test_a_confident_tool_route_constrains_the_provider_to_that_tool(self):
        gov, _ = governor(
            body(action="tool:read_file", action_mass=spread(OPTIONS, "tool:read_file", 0.95))
        )
        host = FakeHost()
        gov.step(host)

        assert len(host.generated) == 1
        assert host.generated[0]["force_tool"] == "read_file"
        assert "read_file" in host.generated[0]["directive"]
        assert gov.last.path == "tool_arguments"

    def test_a_confident_finish_forbids_tools_entirely(self):
        gov, _ = governor(
            body(action="finish", action_mass=spread(OPTIONS, "finish", 0.95), goal=0.99)
        )
        host = FakeHost(tool_calls=[])
        gov.step(host)

        assert host.generated[0]["force_tool"] is None
        assert "Do not call any tool" in host.generated[0]["directive"]
        assert gov.last.path == "final_response"

    def test_the_provider_ignoring_the_selected_tool_blocks_execution(self):
        gov, _ = governor(
            body(action="tool:read_file", action_mass=spread(OPTIONS, "tool:read_file", 0.95))
        )
        # The provider called something else entirely.
        host = FakeHost(tool_calls=["delete_everything"])
        with pytest.raises(JevError, match="ignored the selected tool"):
            gov.step(host)

    def test_the_provider_calling_a_tool_after_a_finish_blocks_execution(self):
        gov, _ = governor(
            body(action="finish", action_mass=spread(OPTIONS, "finish", 0.95), goal=0.99)
        )
        host = FakeHost(tool_calls=["read_file"])
        with pytest.raises(JevError, match="no-tool decision"):
            gov.step(host)


class TestTheGates:
    def test_low_routing_confidence_delegates_instead_of_guessing(self):
        gov, _ = governor(
            body(action="tool:read_file", action_mass=spread(OPTIONS, "tool:read_file", 0.5)),
            confidence_threshold=0.8,
        )
        host = FakeHost()
        gov.step(host)

        assert host.generated[0]["force_tool"] is None
        assert host.generated[0]["directive"] == ""   # unconstrained, as before this package
        assert gov.last.path == "delegated"

    def test_an_explicit_delegate_goes_to_the_model(self):
        gov, _ = governor(
            body(action="delegate", action_mass=spread(OPTIONS, "delegate", 0.99))
        )
        host = FakeHost()
        gov.step(host)
        assert gov.last.path == "delegated"

    def test_a_confident_high_risk_step_stops_and_asks_a_human(self):
        gov, _ = governor(
            body(
                action="tool:read_file",
                action_mass=spread(OPTIONS, "tool:read_file", 0.99),
                risk="high",
            )
        )
        host = FakeHost()
        gov.step(host)

        assert host.generated == []          # nothing was generated
        assert host.said and "stopped" in host.said[0]
        assert gov.last.path == "risk_stop"

    def test_the_risk_stop_is_checked_before_the_confidence_gate(self):
        # A risky action the model is unsure about must not fall through to the generative
        # model, which has fewer brakes than the decision being overridden.
        gov, _ = governor(
            body(
                action="tool:read_file",
                action_mass=spread(OPTIONS, "tool:read_file", 0.40),
                risk="high",
            )
        )
        host = FakeHost()
        gov.step(host)
        assert gov.last.path == "risk_stop"
        assert host.generated == []

    def test_an_unconfident_high_risk_reading_still_delegates(self):
        # Risk `high` but the risk distribution itself is flat: that is not a finding.
        gov, _ = governor(
            body(
                action="tool:read_file",
                action_mass=spread(OPTIONS, "tool:read_file", 0.99),
                risk="high",
                risk_mass={"low": 0.33, "medium": 0.33, "high": 0.34},
            )
        )
        host = FakeHost()
        gov.step(host)
        assert gov.last.path == "delegated"

    def test_a_finish_is_refused_when_completion_is_not_certain_enough(self):
        gov, _ = governor(
            body(action="finish", action_mass=spread(OPTIONS, "finish", 0.99), goal=0.60),
            completion_threshold=0.95,
        )
        host = FakeHost()
        gov.step(host)

        assert gov.last.path == "completion_held"
        assert host.generated[0]["force_tool"] is None
        assert host.generated[0]["directive"] == ""

    def test_clarify_asks_the_user_without_a_generative_call(self):
        gov, _ = governor(body(action="clarify", action_mass=spread(OPTIONS, "clarify", 0.97)))
        host = FakeHost()
        gov.step(host)

        assert host.generated == []
        assert (host.said and "clarify" in host.said[0].lower()) or host.said
        assert gov.last.path == "clarification"


class TestTruncation:
    def test_a_finish_on_a_partial_view_is_refused(self):
        # The dropped history is exactly where unfinished work would be.
        huge = [{"role": "user", "content": "x" * 50_000}] * 6
        gov, _ = governor(
            body(action="finish", action_mass=spread(OPTIONS, "finish", 0.99), goal=0.99),
            state_chars=4_000,
        )
        host = FakeHost(messages=huge)
        gov.step(host)

        assert gov.last.path == "truncated_hold"
        assert gov.last.truncated is True

    def test_a_tool_route_on_a_partial_view_is_still_allowed(self):
        # Routing to a tool from partial evidence is fine; the next turn sees the result.
        huge = [{"role": "user", "content": "x" * 50_000}] * 6
        gov, _ = governor(
            body(action="tool:read_file", action_mass=spread(OPTIONS, "tool:read_file", 0.99)),
            state_chars=4_000,
        )
        host = FakeHost(messages=huge)
        gov.step(host)
        assert gov.last.path == "tool_arguments"


class TestFailurePolicy:
    def test_on_error_delegate_keeps_the_agent_working(self):
        from conftest import FakeResponse

        transport = FakeTransport(FakeResponse(500, {}))
        gov = Governor(JevClient("k", transport=transport), GovernorConfig(on_error="delegate"))
        host = FakeHost()
        gov.step(host)

        assert gov.last.path == "error"
        assert host.generated[0]["force_tool"] is None
        assert host.generated[0]["directive"] == ""

    def test_on_error_stop_raises_rather_than_silently_dropping_the_gate(self):
        from conftest import FakeResponse

        transport = FakeTransport(FakeResponse(500, {}))
        gov = Governor(JevClient("k", transport=transport), GovernorConfig(on_error="stop"))
        with pytest.raises(JevError):
            gov.step(FakeHost())
        assert gov.last.path == "error"


class TestInterrupts:
    def test_an_interrupt_during_the_decision_prevents_any_effect(self):
        gov, _ = governor(
            body(action="tool:read_file", action_mass=spread(OPTIONS, "tool:read_file", 0.99))
        )
        host = FakeHost(interrupted=True)
        with pytest.raises(InterruptedError):
            gov.step(host)
        assert host.generated == []
        assert host.said == []


class TestStaticTools:
    def test_a_configured_payload_runs_with_no_generative_call(self):
        gov, _ = governor(
            body(action="tool:read_file", action_mass=spread(OPTIONS, "tool:read_file", 0.99)),
            static_tools={"read_file": {"path": "notes.txt"}},
        )
        host = FakeHost()
        gov.step(host)

        assert host.generated == []
        assert host.static_calls == [("read_file", {"path": "notes.txt"})]
        assert gov.last.path == "static_tool"

    def test_a_payload_that_does_not_fit_the_schema_is_refused(self):
        gov, _ = governor(
            body(action="tool:read_file", action_mass=spread(OPTIONS, "tool:read_file", 0.99)),
            static_tools={"read_file": {"wrong_field": 1}},
        )
        with pytest.raises(JevError, match="do not match the tool schema"):
            gov.step(FakeHost())

    def test_a_static_payload_is_not_used_for_a_risky_step(self):
        gov, _ = governor(
            body(
                action="tool:read_file",
                action_mass=spread(OPTIONS, "tool:read_file", 0.99),
                risk="medium",
            ),
            static_tools={"read_file": {"path": "notes.txt"}},
        )
        host = FakeHost()
        gov.step(host)

        assert host.static_calls == []
        assert gov.last.path == "tool_arguments"


class TestTelemetry:
    def test_every_recorded_path_is_a_declared_one(self):
        from jev_governor import PATHS

        gov, _ = governor(
            body(action="delegate", action_mass=spread(OPTIONS, "delegate", 0.99))
        )
        gov.step(FakeHost())
        assert gov.last.path in PATHS

    def test_the_event_log_does_not_grow_without_bound(self):
        from conftest import FakeResponse
        from jev_governor import MAX_EVENTS

        payload = body(action="delegate", action_mass=spread(OPTIONS, "delegate", 0.99))
        count = MAX_EVENTS + 25
        transport = FakeTransport(*[FakeResponse(200, payload) for _ in range(count)])
        gov = Governor(JevClient("k", transport=transport))
        host = FakeHost()
        for _ in range(count):
            gov.step(host)
        assert len(gov.events) == MAX_EVENTS

    def test_an_event_carries_the_resolved_model_so_a_change_is_attributable(self):
        gov, _ = governor(
            body(
                action="delegate",
                action_mass=spread(OPTIONS, "delegate", 0.99),
                model="jev-1.14.0",
            )
        )
        gov.step(FakeHost())
        assert gov.last.model == "jev-1.14.0"


class TestTheRequest:
    def test_the_transcript_is_never_mutated(self):
        gov, _ = governor(
            body(action="tool:read_file", action_mass=spread(OPTIONS, "tool:read_file", 0.99))
        )
        messages = [
            {"role": "system", "content": "be careful"},
            {"role": "user", "content": "read notes.txt"},
        ]
        before = [dict(m) for m in messages]
        gov.step(FakeHost(messages=messages))
        assert messages == before

    def test_the_ballot_contains_the_hosts_real_tools_and_the_controls(self):
        gov, transport = governor(
            body(action="tool:read_file", action_mass=spread(OPTIONS, "tool:read_file", 0.99))
        )
        gov.step(FakeHost())
        criteria = transport.requests[0]["payload"]["questions"]["action"]["criteria"]
        assert set(criteria) == {"finish", "clarify", "delegate", "tool:read_file"}

    def test_the_api_key_travels_as_a_bearer_header_and_not_in_the_body(self):
        import json as _json

        gov, transport = governor(
            body(action="delegate", action_mass=spread(OPTIONS, "delegate", 0.99))
        )
        gov.step(FakeHost())
        request = transport.requests[0]
        assert request["headers"]["Authorization"] == "Bearer test-key"
        assert "test-key" not in _json.dumps(request["payload"])
