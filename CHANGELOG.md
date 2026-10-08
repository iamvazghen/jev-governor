# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-10-08

First release. Pre-1.0: the `Host` protocol and `GovernorConfig` fields may still change, and
breaking changes will be called out here.

### Added

- **`Governor`** — a seven-guard decision loop over one System One call per step. Nine recorded
  paths (`static_tool`, `tool_arguments`, `final_response`, `clarification`, `risk_stop`,
  `delegated`, `completion_held`, `truncated_hold`, `error`), each with the probabilities, the
  resolved model id, and the decision latency behind it.
- **Enforcement, not advice.** The host constrains the provider to the selected tool, and the
  result is checked against it — a provider that ignores `tool_choice` raises before anything
  executes. The inverse also holds: a tool call after a `finish` decision is blocked.
- **An independent completion gate.** `goal_met` is a separate question in the same request, so
  it cannot see the `action` answer; a `finish` below `completion_threshold` is refused.
- **A truncation hold.** A `finish` decided on a partial view of the transcript is refused,
  because the dropped history is where unfinished work would be.
- **A risk gate ahead of the confidence gate**, so uncertainty about risk never falls through to
  the generative model.
- **A distrust layer** (`decision.py`): rejects non-finite and `bool` probabilities, a
  distribution over the wrong option set or one that does not sum to one, and a choice that is
  not its own argmax; clamps reported confidence to the winning option's own mass.
- **A bounded state projection** (`state.py`) enforced on *serialized* length, built fresh per
  step, which never mutates or reorders the canonical transcript.
- **`JevClient` / `AsyncJevClient`** — one retry for `429`/`529` only, distinguishable `401`
  and `403` diagnoses, injectable transport, and no credential or response body in any
  exception message.
- **`GovernorConfig`** — validated at construction, with unknown keys an error rather than a
  warning.
- **`static_tools`** — zero-generative-cost execution for fixed payloads, validated against the
  tool's live JSON schema, remote `$ref` refused, and only on a `low` risk verdict.
- **Adapters**: `openai_chat` (any OpenAI-compatible loop), `hermes` (a six-line seam),
  `pydantic_ai` (`GovernedModel`, a `WrapperModel` that constrains via `function_tools` and
  `allow_text_output`), and `openclaw` (`OpenClawRouter` — fleet routing with `argv`-only
  dispatch, the message in a file, no `--deliver`, and an environment allowlist built from
  nothing).
- 210 hermetic tests — no network, no API key, no provider. `scripts/verify_live.py` covers the
  live path on demand.
- Documentation: architecture and the reasoning for each guard's position, Hermes install,
  limits, comparison, roadmap, and a guide to writing an adapter.

### Notes on provenance

This supersedes an earlier approach that forked Hermes Agent to add roughly 670 lines. The fork
meant every upstream release had to be merged rather than installed alongside, the decision
layer could not be tested without all of Hermes, and users of other frameworks could not use
any of it. This package contains **no Hermes source** and integrates through a six-method host
protocol instead, so the decision layer and the agent framework version independently.

### Known gaps

Carried from [docs/limits.md](docs/limits.md) so they are visible in release notes:

- Streaming is not governed; `request_stream` falls through to the wrapped model.
- `Governor` is synchronous, so the Pydantic AI adapter runs it on a worker thread.
- The Pydantic AI, Hermes and OpenClaw adapters are **not verified in CI against live
  frameworks or a live gateway** — their logic and transport properties are tested, real round
  trips are not.
- Vendor latency and price figures are not reproduced here as measurements.

[0.1.0]: https://github.com/iamvazghen/jev-governor/releases/tag/v0.1.0
