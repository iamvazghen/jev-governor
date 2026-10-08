"""The distrust layer. Each test here is a way a typed answer can still be wrong."""

from __future__ import annotations

import dataclasses

import pytest

from conftest import answer_choice, answer_noul, body
from jev_governor import Decision, JevError, choice, noul, probability

OPTIONS = {"finish", "clarify", "delegate", "tool:read_file"}


class TestProbability:
    def test_accepts_the_closed_unit_interval(self):
        assert probability(0) == 0.0
        assert probability(1) == 1.0
        assert probability(0.37) == pytest.approx(0.37)

    def test_rejects_bool_even_though_it_is_an_int(self):
        # isinstance(True, int) is True in Python, so without an explicit check a stray
        # boolean would read as full certainty.
        with pytest.raises(JevError):
            probability(True)
        with pytest.raises(JevError):
            probability(False)

    @pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("inf"), "0.9", None, []])
    def test_rejects_everything_else(self, value):
        with pytest.raises(JevError):
            probability(value)


class TestChoice:
    def test_returns_the_selection_and_a_clamped_confidence(self):
        answer = answer_choice("finish", {"finish": 0.7, "clarify": 0.3}, confidence=0.99)
        selected, confidence = choice(answer, {"finish", "clarify"})
        assert selected == "finish"
        # The service claimed 0.99 but only 0.7 of the mass is on the winner.
        assert confidence == pytest.approx(0.7)

    def test_keeps_a_confidence_below_the_winning_mass(self):
        answer = answer_choice("finish", {"finish": 0.9, "clarify": 0.1}, confidence=0.55)
        _, confidence = choice(answer, {"finish", "clarify"})
        assert confidence == pytest.approx(0.55)

    def test_rejects_an_option_that_was_never_offered(self):
        answer = answer_choice("tool:rm_rf", {"tool:rm_rf": 1.0})
        with pytest.raises(JevError, match="not offered"):
            choice(answer, OPTIONS)

    def test_rejects_a_distribution_over_a_different_option_set(self):
        answer = answer_choice("finish", {"finish": 0.6, "something_else": 0.4})
        with pytest.raises(JevError, match="different option set"):
            choice(answer, {"finish", "clarify"})

    def test_rejects_a_distribution_that_does_not_sum_to_one(self):
        answer = answer_choice("finish", {"finish": 0.2, "clarify": 0.2})
        with pytest.raises(JevError, match="sum to one"):
            choice(answer, {"finish", "clarify"})

    def test_rejects_a_choice_that_is_not_its_own_argmax(self):
        answer = {
            "type": "choice",
            "choice": "clarify",
            "probabilities": {"finish": 0.8, "clarify": 0.2},
            "confidence": 0.8,
        }
        with pytest.raises(JevError, match="disagrees"):
            choice(answer, {"finish", "clarify"})

    def test_tolerates_a_floating_point_tie(self):
        third = 1.0 / 3.0
        answer = answer_choice("a", {"a": third, "b": third, "c": third})
        selected, _ = choice(answer, {"a", "b", "c"})
        assert selected == "a"

    def test_rejects_the_wrong_primitive(self):
        with pytest.raises(JevError, match="wrong primitive"):
            choice(answer_noul(0.9), {"finish"})

    def test_rejects_a_non_object(self):
        with pytest.raises(JevError):
            choice("finish", {"finish"})


class TestNoul:
    def test_returns_the_probability(self):
        assert noul(answer_noul(0.82)) == pytest.approx(0.82)

    def test_rejects_the_wrong_primitive(self):
        with pytest.raises(JevError, match="wrong primitive"):
            noul(answer_choice("finish", {"finish": 1.0}))


class TestDecisionParse:
    def test_parses_a_well_formed_response(self):
        decision = Decision.parse(body(action="finish", goal=0.97), OPTIONS)
        assert decision.action == "finish"
        assert decision.goal_probability == pytest.approx(0.97)
        assert decision.risk == "low"
        assert decision.model == "jev-1.13.0"
        assert decision.usage["input_tokens"] == 120

    def test_exposes_the_tool_name_only_for_tool_actions(self):
        assert Decision.parse(body(action="tool:read_file"), OPTIONS).tool == "read_file"
        assert Decision.parse(body(action="finish"), OPTIONS).tool is None

    def test_rejects_a_missing_resolved_model_id(self):
        payload = body()
        payload["model"] = ""
        with pytest.raises(JevError, match="resolved model id"):
            Decision.parse(payload, OPTIONS)

    def test_rejects_a_bool_token_count(self):
        payload = body()
        payload["usage"]["input_tokens"] = True
        with pytest.raises(JevError, match="usage metadata"):
            Decision.parse(payload, OPTIONS)

    def test_rejects_a_negative_token_count(self):
        payload = body()
        payload["usage"]["output_tokens"] = -1
        with pytest.raises(JevError, match="usage metadata"):
            Decision.parse(payload, OPTIONS)

    @pytest.mark.parametrize("missing", ["action", "goal_met", "risk"])
    def test_rejects_a_missing_answer(self, missing):
        payload = body()
        del payload["answers"][missing]
        with pytest.raises(JevError, match="Malformed"):
            Decision.parse(payload, OPTIONS)

    def test_rejects_a_non_object_response(self):
        with pytest.raises(JevError):
            Decision.parse("ok", OPTIONS)

    def test_is_immutable(self):
        decision = Decision.parse(body(), OPTIONS)
        with pytest.raises(dataclasses.FrozenInstanceError):
            decision.action = "finish"  # type: ignore[misc]
