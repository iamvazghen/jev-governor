"""Config validation happens once, at startup, with a message that names the field."""

from __future__ import annotations

import dataclasses

import pytest

from jev_governor import ON_ERROR_POLICIES, GovernorConfig, JevError
from jev_governor.config import MAX_STATE_CHARS, MIN_STATE_CHARS


class TestDefaults:
    def test_the_defaults_are_usable_and_sane(self):
        config = GovernorConfig()
        assert 0 < config.confidence_threshold <= 1
        assert config.completion_threshold >= config.confidence_threshold
        assert config.on_error in ON_ERROR_POLICIES

    def test_completion_is_stricter_than_routing_by_default(self):
        # The failure modes are not symmetric: a needless delegation costs cents, while
        # stopping early reports success on work that never happened.
        config = GovernorConfig()
        assert config.completion_threshold > config.confidence_threshold

    def test_is_immutable(self):
        config = GovernorConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.confidence_threshold = 0.1  # type: ignore[misc]

    def test_static_tools_default_is_not_shared_between_instances(self):
        first = GovernorConfig()
        first.static_tools["x"] = {}
        assert GovernorConfig().static_tools == {}


class TestThresholds:
    @pytest.mark.parametrize("value", [-0.1, 1.1, "0.8", None, True])
    def test_an_invalid_confidence_threshold_is_refused(self, value):
        with pytest.raises(JevError):
            GovernorConfig(confidence_threshold=value)

    @pytest.mark.parametrize("value", [-0.1, 1.1, "0.95", None])
    def test_an_invalid_completion_threshold_is_refused(self, value):
        with pytest.raises(JevError):
            GovernorConfig(completion_threshold=value)


class TestStateChars:
    @pytest.mark.parametrize("value", [MIN_STATE_CHARS, 24_000, MAX_STATE_CHARS])
    def test_accepts_the_documented_range(self, value):
        assert GovernorConfig(state_chars=value).state_chars == value

    @pytest.mark.parametrize("value", [0, 100, MIN_STATE_CHARS - 1, MAX_STATE_CHARS + 1, 1.5, True])
    def test_refuses_anything_outside_it(self, value):
        with pytest.raises(JevError, match="state_chars"):
            GovernorConfig(state_chars=value)


class TestTimeout:
    @pytest.mark.parametrize("value", [0, -1, 31, True, "5"])
    def test_refuses_an_unusable_timeout(self, value):
        with pytest.raises(JevError, match="timeout"):
            GovernorConfig(timeout=value)

    def test_accepts_a_fractional_timeout(self):
        assert GovernorConfig(timeout=0.4).timeout == 0.4


class TestPolicy:
    @pytest.mark.parametrize("value", ON_ERROR_POLICIES)
    def test_accepts_each_declared_policy(self, value):
        assert GovernorConfig(on_error=value).on_error == value

    @pytest.mark.parametrize("value", ["llm", "retry", "", None])
    def test_refuses_an_undeclared_policy(self, value):
        with pytest.raises(JevError, match="on_error"):
            GovernorConfig(on_error=value)

    @pytest.mark.parametrize("value", ["", None, 5])
    def test_refuses_an_empty_model_id(self, value):
        with pytest.raises(JevError, match="model"):
            GovernorConfig(model=value)


class TestStaticTools:
    def test_accepts_a_mapping_of_names_to_argument_objects(self):
        config = GovernorConfig(static_tools={"docker_ps": {}, "read": {"path": "a"}})
        assert set(config.static_tools) == {"docker_ps", "read"}

    @pytest.mark.parametrize("value", [["read"], "read", 5])
    def test_refuses_a_non_mapping(self, value):
        with pytest.raises(JevError, match="static_tools"):
            GovernorConfig(static_tools=value)

    def test_refuses_a_non_object_payload(self):
        with pytest.raises(JevError, match="argument object"):
            GovernorConfig(static_tools={"read": "notes.txt"})


class TestFromDict:
    def test_builds_from_a_mapping(self):
        config = GovernorConfig.from_dict({"confidence_threshold": 0.9, "on_error": "stop"})
        assert config.confidence_threshold == 0.9
        assert config.on_error == "stop"

    def test_ignores_the_enabled_flag_that_host_wiring_consumes(self):
        assert GovernorConfig.from_dict({"enabled": True}).on_error == "delegate"

    def test_an_unknown_key_is_an_error_not_a_warning(self):
        # A silently-ignored typo leaves the gate at its default while the operator believes
        # it was tightened, and nothing in the logs says otherwise.
        with pytest.raises(JevError, match="confidence_threshhold"):
            GovernorConfig.from_dict({"confidence_threshhold": 0.9})

    def test_names_every_unknown_key(self):
        with pytest.raises(JevError) as caught:
            GovernorConfig.from_dict({"alpha": 1, "beta": 2})
        assert "alpha" in str(caught.value) and "beta" in str(caught.value)

    def test_refuses_a_non_mapping(self):
        with pytest.raises(JevError, match="object"):
            GovernorConfig.from_dict(["confidence_threshold"])
