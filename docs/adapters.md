# Writing an adapter

A `Host` is six methods. The package never imports your framework and you never import the
decision logic — `jev_governor.host.Host` is a `Protocol`, so a plain class that has the
methods is a valid host.

```python
class MyHost:
    def tools(self) -> list[dict]: ...
    def messages(self) -> list[dict]: ...
    def generate(self, *, force_tool: str | None, directive: str): ...
    def called_tools(self, response) -> list[str]: ...
    def say(self, text: str): ...
    def say_tool_call(self, name: str, arguments: dict): ...
    def interrupted(self) -> bool: ...
```

(That is seven including `interrupted`, which can always return `False`.)

## Read this before you start

**The only hard requirement is that `generate` can genuinely force a tool choice, and
`called_tools` can report what actually happened.** Everything else is data plumbing.

If your framework cannot constrain the model to one named tool, you *can* still write an
adapter — but say so loudly in its docstring, because the governor's central property is gone.
A decision layer that can only suggest is advice, and advice is worse than nothing here: it
reads as a guarantee while providing none. The OpenClaw adapter is the honest version of that
situation — it does not pretend to govern the inner loop, it governs the one decision it can
actually enforce.

## `tools()`

Return OpenAI-style schemas. Both shapes are accepted:

```python
{"type": "function", "function": {"name": "read_file", "description": "...", "parameters": {...}}}
{"name": "read_file", "description": "...", "parameters": {...}}
```

Called fresh on every decision, so a context-dependent toolset stays correct.

**Descriptions are the whole ballot.** They become `choice` criteria verbatim. A tool described
as `"personal"` routes badly; `"Household bills, appointments, personal errands"` routes well.
If your framework's descriptions are thin, that is the first thing to fix — no threshold tuning
compensates for an unroutable ballot.

## `messages()`

Oldest-first, OpenAI message form:

```python
{"role": "system" | "user" | "assistant" | "tool", "content": str | list, "name": str, "tool_calls": [...]}
```

Only read, never mutated. If your framework has its own message objects, project them — see
`adapters/pydantic_ai.py:_messages` for a worked example that flattens parts by kind.

## `generate(force_tool, directive)`

The contract:

| `force_tool` | `directive` | What you must do |
|---|---|---|
| `"read_file"` | non-empty | Constrain the provider to exactly one call of `read_file` |
| `None` | non-empty | Forbid tool calls entirely — prose only |
| `None` | `""` | Nothing: plain unconstrained generation, exactly as before |

That third row is the delegation path and it must be a true passthrough. If your adapter adds
anything there, the fallback stops being equivalent to the un-governed agent and
`on_error: delegate` no longer means what it says.

**Append the directive to a copy.** Mutating the caller's message list changes its identity and
invalidates the provider's prompt-cache prefix — which costs more than this package saves.

Two mechanisms, as examples:

```python
# OpenAI-compatible
request["tool_choice"] = {"type": "function", "function": {"name": force_tool}}
request["parallel_tool_calls"] = False

# Pydantic AI: no tool_choice field, so narrow the request parameters instead
dataclasses.replace(parameters, function_tools=[chosen], allow_text_output=False)
```

## `called_tools(response)`

Return the names actually called, empty list for prose. This is what makes enforcement real:

```python
if called != [force_tool]:
    raise JevError("The provider ignored the selected tool; execution blocked")
```

Use your framework's own response normaliser if it has one (the Hermes adapter does), so you
see what the agent will see rather than a second interpretation of the same bytes.

## `say(text)` and `say_tool_call(name, arguments)`

Build a native response **locally, with no model call.** `say` for the risk stop and the
clarification; `say_tool_call` for a validated `static_tools` payload. If these round-trip to a
provider, three of the nine paths quietly start costing money.

Stamp them with an identifiable model id (`"jev-governor-local"`) so they are never mistaken
for provider output in a log.

## `interrupted()`

`False` is fine if your framework has no such concept. If it cancels tasks itself (Pydantic AI)
there is nothing to consult. If it sets a flag (Hermes), read it — a decision takes long enough
for a Ctrl-C to arrive, and this is the last moment before an effect.

## Testing it

Copy `tests/conftest.py`. The useful shape is a fake transport returning canned response
bodies plus a host that records what it was asked to do, then assert on `governor.last.path`.

Four tests worth having for any adapter:

1. A confident tool route reaches the provider **with the constraint set**.
2. A provider that ignores the constraint **raises**.
3. Delegation is a **true passthrough** — no directive, no `tool_choice`.
4. `say` makes **no network call**.

Number two is the one people skip, and it is the one that proves the adapter works.

## Contributing it

Open a PR with the adapter, its tests, and a docstring that names what it cannot do. An adapter
honest about its limits is more useful than one that implies a guarantee it cannot keep.
