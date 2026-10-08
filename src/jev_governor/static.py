"""Validating a static tool payload against the tool's real schema.

A ``static_tools`` entry is trusted configuration: an operator wrote it, so it does not need a
generative model to produce it, and skipping that call is the entire point. "Trusted" is not
the same as "correct", though, and the two ways a trusted payload goes wrong are both quiet:

* The tool's schema changed -- a renamed field, a new required one -- and the configured
  payload now describes a call that no longer exists. Without a check, that surfaces as a
  confusing tool error attributed to the agent rather than to the config.
* The payload was always wrong and simply never exercised, because this path only triggers on
  a specific route that happens rarely.

So the payload is validated against the schema the host is advertising *right now*, in the
same step that will execute it. ``jsonschema`` is an optional dependency: without it, static
execution refuses rather than proceeding unvalidated, because an unvalidated shortcut past the
generative model is worse than no shortcut.

Remote ``$ref`` resolution is refused outright. A schema that fetches part of itself over the
network turns argument validation into an HTTP request to somewhere this package cannot vouch
for, on a path whose purpose is to avoid asking anything.
"""

from __future__ import annotations

import json
from typing import Any

from .errors import JevError
from .questions import tool_spec


def schema_for(tools: list[dict], name: str) -> dict:
    """Return the JSON schema for the parameters of tool ``name`` as the host advertises it."""
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        spec = tool_spec(tool)
        if spec.get("name") == name:
            parameters = spec.get("parameters")
            return parameters if isinstance(parameters, dict) else {"type": "object"}
    raise JevError(f"Static tool {name!r} is not in the host's current toolset")


def validate_static_arguments(tools: list[dict], name: str, arguments: Any) -> None:
    """Raise :class:`~jev_governor.errors.JevError` unless ``arguments`` fit the tool schema."""
    if not isinstance(arguments, dict):
        raise JevError(f"Static arguments for {name!r} must be an object")
    schema = schema_for(tools, name)

    try:
        from jsonschema import Draft202012Validator, SchemaError
    except ImportError:
        raise JevError(
            "Static tool execution needs the jsonschema extra "
            "(pip install 'jev-governor[static]'); use dynamic arguments instead"
        ) from None

    # Checked on the serialized schema so a `$ref` nested at any depth is caught.
    if '"$ref"' in json.dumps(schema):
        raise JevError(
            f"The schema for {name!r} uses $ref; static tool schemas must be self-contained"
        )

    try:
        Draft202012Validator.check_schema(schema)
        valid = Draft202012Validator(schema).is_valid(arguments)
    except (SchemaError, TypeError, ValueError) as exc:
        raise JevError(f"The host advertises an invalid schema for {name!r}") from exc

    if not valid:
        raise JevError(f"Static arguments for {name!r} do not match the tool schema")
