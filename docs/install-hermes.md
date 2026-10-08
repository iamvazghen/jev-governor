# Installing into Hermes Agent

Hermes funnels every provider call through one function, so this is a **seam, not a fork**. Two
files change, six lines in total. That matters for the reason below.

## Why not fork Hermes

An earlier version of this work *was* a fork — the whole upstream tree carried along to add
about 670 lines. It worked, and it was the wrong shape:

- Every upstream release had to be **merged** rather than installed alongside, with conflict
  resolution in files the integration never touched.
- The decision layer could not be tested without standing up all of Hermes.
- Nobody running Pydantic AI or anything else could use any of it.

A standalone package with a six-method host protocol lets the two sides be versioned, tested
and upgraded independently. You `pip install` this, you keep tracking Hermes `main` normally,
and the only thing you maintain is the seam below.

## Step 1 — install

```bash
pip install jev-governor
pip install "jev-governor[static]"   # only if you want static_tools
```

## Step 2 — the seam in `agent/turn_api_call.py`

Find `_perform_api_call` inside `perform_api_call`. Rename it to `_generate_api_call` and put a
new `_perform_api_call` in front of it:

```python
    def _perform_api_call(next_api_kwargs):
        governor = getattr(agent, "_jev_governor", None)
        if governor is not None:
            return governor.call(agent, next_api_kwargs, _generate_api_call)
        return _generate_api_call(next_api_kwargs)

    def _generate_api_call(next_api_kwargs):
        # ... the original body, unchanged ...
```

Nothing else in that file moves. When the governor is absent — not configured, or disabled for
this profile — the second function runs directly and the call path is byte-for-byte what it was.

## Step 3 — the wiring in `agent/agent_init.py`

In `init_agent`, after `_agent_cfg` is loaded:

```python
    from jev_governor.adapters.hermes import install
    install(agent, _agent_cfg)
```

`install` reads the `jev` section. Absent, or `enabled: false`, and it sets
`agent._jev_governor = None` — so switching the feature off is one line of YAML rather than an
uninstall, and the seam above falls back to plain generation.

## Step 4 — configure per profile

```yaml
jev:
  enabled: true
  confidence_threshold: 0.80   # below this, delegate rather than route
  completion_threshold: 0.95   # below this, refuse to stop
  state_chars: 24000
  timeout: 5
  on_error: delegate           # or: stop
  goal_hint: "You operate a home lab and a small business."
  static_tools:
    docker_ps: {}
```

An unknown key here is an **error**, not a warning — a typo that is silently ignored leaves the
gate at its default while you believe you tightened it.

## Step 5 — the credential

`install` reads `TYPESAFE_API_KEY` from Hermes's own secret scope
(`agent.secret_scope`) when it is available, and falls back to the process environment. It is
never read from this package's config, because a credential should have exactly one home and
Hermes already provides one.

## Step 6 — verify

```bash
pytest                                   # Hermes's own suite: unchanged
python -m jev_governor.selfcheck         # not a thing; use the script below
python scripts/verify_live.py            # from this repo, with your key
```

Then run a real turn and look at the events:

```python
for event in agent._jev_governor.events[-5:]:
    print(event.path, event.action, round(event.confidence or 0, 3), event.decision_ms)
```

You want to see a mix. All `delegated` means the thresholds are too tight or the tool
descriptions are too thin to route on. All `tool_arguments` with no `delegated` at all is worth
a second look — a decision layer that is never unsure is usually one that is not being asked
hard questions.

## What stays upstream

Tools, execution sandboxes, session persistence, memory and FTS/vector retrieval, skills,
context compression, and every provider transport remain untouched Hermes code. This package
reads the toolset and the transcript and constrains one provider call. That is the entire
contact surface.

## Keeping current with upstream

Because this is an installed dependency rather than a fork, the normal flow works:

```bash
git fetch upstream && git merge upstream/main
```

The only file that can conflict is `turn_api_call.py`, and only if upstream restructures
`_perform_api_call` itself. If that happens, the fix is to re-apply the same two-function
pattern around whatever the new call site is called — the governor does not care what the
surrounding function is named, only that it is handed the request and a way to send it.

## Limitations

- **`api_mode` must be `chat_completions`.** `codex_responses` is a different request shape
  with different tool semantics; `HermesHost` raises rather than mis-parsing it.
- **Streaming is not governed.** Check how `_should_stream` resolves for your profile.
- This adapter is **not verified in CI against a live Hermes checkout** — the seam is small
  enough to read, but verify it in your own fork and please report breakage.
