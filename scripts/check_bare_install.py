#!/usr/bin/env python
"""Prove the package is usable with none of the optional extras installed.

Run by CI in an environment that has only ``httpx``. Two properties, both of which are easy to
break by moving an import to the top of a file:

1. **Every module imports.** The adapters import their frameworks lazily, inside functions, so
   that an uninstalled framework is an error when you *call* it -- with a message naming the
   extra -- rather than an ImportError when you merely import this package. Hoisting one of
   those imports would make installing `jev-governor` require pydantic-ai.

2. **Static execution refuses rather than proceeding unvalidated.** Without ``jsonschema``
   there is no way to check a configured payload against the live tool schema, and an
   unvalidated shortcut past the generative model is worse than no shortcut. It must raise, and
   the message must name the missing extra.
"""

from __future__ import annotations

import importlib
import sys

MODULES = (
    "jev_governor",
    "jev_governor.client",
    "jev_governor.config",
    "jev_governor.decision",
    "jev_governor.errors",
    "jev_governor.governor",
    "jev_governor.host",
    "jev_governor.questions",
    "jev_governor.state",
    "jev_governor.static",
    "jev_governor.adapters.openai_chat",
    "jev_governor.adapters.hermes",
    "jev_governor.adapters.openclaw",
    "jev_governor.adapters.pydantic_ai",
)


def main() -> int:
    for name in MODULES:
        importlib.import_module(name)
        print(f"  imported {name}")

    from jev_governor.errors import JevError
    from jev_governor.static import validate_static_arguments

    try:
        import jsonschema  # noqa: F401
    except ImportError:
        pass
    else:
        print("\njsonschema is installed, so the refusal path cannot be checked here.")
        print("This job is meant to run with no optional extras.")
        return 1

    tools = [{"name": "t", "parameters": {"type": "object"}}]
    try:
        validate_static_arguments(tools, "t", {})
    except JevError as error:
        if "jsonschema" not in str(error):
            print(f"\nRefused, but the message does not name the extra: {error}")
            return 1
        print(f"\n  static execution refuses without jsonschema: {error}")
    else:
        print("\nStatic validation silently proceeded with no schema validator available.")
        return 1

    print("\nBare install is sound.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
