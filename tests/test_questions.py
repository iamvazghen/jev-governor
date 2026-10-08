"""The question set is the architecture, so this is where it is pinned down."""

from __future__ import annotations

import pytest

from conftest import _tool
from jev_governor import CONTROL_OPTIONS, MAX_CRITERIA, JevError, build_questions
from jev_governor.questions import MAX_DESCRIPTION_CHARS, tool_options


class TestShape:
    def test_asks_exactly_three_questions(self):
        questions = build_questions([_tool("read_file", "Read a file.")])
        assert set(questions) == {"action", "goal_met", "risk"}

    def test_uses_the_documented_primitives(self):
        questions = build_questions([])
        assert questions["action"]["type"] == "choice"
        assert questions["goal_met"]["type"] == "noul"
        assert questions["risk"]["type"] == "choice"

    def test_a_noul_question_carries_no_criteria(self):
        # A noul's criteria is optional and the probability is the answer; sending a criteria
        # object for one is how a 422 happens.
        assert "criteria" not in build_questions([])["goal_met"]

    def test_risk_offers_exactly_three_levels(self):
        assert set(build_questions([])["risk"]["criteria"]) == {"low", "medium", "high"}


class TestTheBallot:
    def test_the_controls_are_always_present(self):
        criteria = build_questions([])["action"]["criteria"]
        assert set(criteria) == set(CONTROL_OPTIONS)

    def test_tools_are_namespaced_so_they_cannot_shadow_a_control(self):
        criteria = build_questions([_tool("read_file", "Read a file.")])["action"]["criteria"]
        assert "tool:read_file" in criteria
        assert set(CONTROL_OPTIONS) <= set(criteria)

    def test_a_tool_named_like_a_control_is_refused(self):
        with pytest.raises(JevError, match="reserved control option"):
            tool_options([_tool("finish", "Finish up.")])

    def test_a_duplicate_tool_name_is_refused(self):
        with pytest.raises(JevError, match="Duplicate"):
            tool_options([_tool("read_file", "A."), _tool("read_file", "B.")])

    def test_a_tool_without_a_name_is_refused(self):
        with pytest.raises(JevError, match="missing its name"):
            tool_options([{"type": "function", "function": {"description": "nameless"}}])

    def test_accepts_the_bare_tool_form_as_well_as_the_nested_one(self):
        options = tool_options([{"name": "ping", "description": "Ping a host."}])
        assert options == {"tool:ping": "Ping a host."}

    def test_an_undescribed_tool_says_so_instead_of_guessing(self):
        options = tool_options([{"name": "zz", "description": ""}])
        assert "No description was supplied" in options["tool:zz"]

    def test_a_long_description_is_truncated_not_rejected(self):
        options = tool_options([_tool("big", "d" * 5_000)])
        assert len(options["tool:big"]) == MAX_DESCRIPTION_CHARS

    def test_too_many_tools_is_refused_with_the_actual_advice(self):
        many = [_tool(f"t{index}", f"Tool {index}.") for index in range(MAX_CRITERIA)]
        with pytest.raises(JevError, match="split the toolset across more agents"):
            build_questions(many)

    def test_the_ceiling_counts_the_controls(self):
        # MAX_CRITERIA total, three of which are always controls.
        room = MAX_CRITERIA - len(CONTROL_OPTIONS)
        just_fits = [_tool(f"t{index}", "x") for index in range(room)]
        assert len(build_questions(just_fits)["action"]["criteria"]) == MAX_CRITERIA


class TestPromptInjectionStance:
    @pytest.mark.parametrize("name", ["action", "goal_met", "risk"])
    def test_every_question_declares_tool_output_to_be_evidence_not_instructions(self, name):
        instructions = build_questions([])[name]["instructions"]
        assert "never instructions" in instructions

    def test_risk_says_it_is_not_authorization(self):
        assert "not authorization" in build_questions([])["risk"]["instructions"]

    def test_completion_rejects_a_plan_as_evidence_of_completion(self):
        instructions = build_questions([])["goal_met"]["instructions"]
        assert "plan" in instructions and "not complete" in instructions


class TestGoalHint:
    def test_is_appended_when_supplied(self):
        questions = build_questions([], goal_hint="You run a home lab.")
        assert "You run a home lab." in questions["action"]["instructions"]

    def test_is_absent_when_not_supplied(self):
        assert "Operating context" not in build_questions([])["action"]["instructions"]
