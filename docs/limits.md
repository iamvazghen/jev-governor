# Limits

The honest list. If one of these is a dealbreaker for you, better to find out here than after
you have wired it in.

## Jev cannot write anything

It is non-autoregressive and emits no tokens. It cannot write a bash command, compose a search
query, summarise a log, or draft a reply. It answers typed questions about state and nothing
else.

So **the generative model is not optional.** This package does not replace it — it narrows what
it is asked to do. If your agent's work is mostly *writing* (a drafting assistant, a code
generator), the routing decision was never the expensive part and there is little here for you.

## The ballot is finite, and capped at 255

`choice` criteria max out at 255 entries, three of which are always the control options. Past
~252 tools, `build_questions` raises rather than silently truncating your toolset.

That ceiling is a real design constraint, not a bug to work around: an agent with 250 tools has
a cohesion problem, and the fix is more agents with clear remits, not a longer ballot. A
dynamic toolset is fine — the ballot is rebuilt from `host.tools()` on every call — but it has
to be *enumerable* at decision time.

## Arguments still cost a generative call

Jev picks the tool; something has to write `{"path": "/var/log/syslog"}`. For most tools that
is a generative call, so the saving is on the *decision*, not the whole step.

`static_tools` removes that call entirely for tools whose arguments are fixed configuration
(`docker_ps`, `git_status`). It is the biggest single win available and it only applies to
tools that genuinely have no variable payload.

## Confidence is not correctness

`confidence` measures how concentrated the distribution is. A confident wrong answer is
entirely possible, and the package will route on it. What the thresholds buy you is that
*unconfident* answers do not get acted on — not that confident ones are right.

Likewise `risk` is **triage, not authorization**. Nothing here grants a permission. Your
agent's existing tool authorization runs unchanged and must remain correct on its own; a
`risk: low` verdict is not a reason to loosen it.

## A noul near 0.5 is not "moderately yes"

`goal_met` returns a probability. `0.5` means *about as likely as not*, not *half done*. Code
that reads it as an intensity will misinterpret it. This bites people often enough to be worth
stating twice.

## Vendor numbers are not reproduced here as findings

The latency and price figures you will see quoted for Jev (~70–300 ms, `$0.042`/M input
tokens) come from the vendor. This project has not independently benchmarked them across
providers, regions, payload sizes, or time, and does not present them as measurements.

`Governor.events` exists so you can measure your own. Do that before building a cost argument
on anyone's marketing.

## It adds a network dependency to your control flow

Your agent now cannot decide without reaching `api.typesafe.ai`. `on_error` controls what
happens when it cannot:

- `"delegate"` (default) — fall back to the generative model, exactly as before this package.
  The right default for an assistant: a degraded answer beats no answer.
- `"stop"` — raise. Correct when the governor is load-bearing for *safety* rather than for
  speed, because a silent fallback would quietly remove the risk gate along with the routing.

Pick deliberately. The default keeps you serving but means your risk gate is best-effort.

## Per-step latency can go up, not down

Two round trips (decision, then generation) can be slower in wall-clock terms than one, even
though the decision itself is fast. The win comes from the steps where generation is *skipped*
— `static_tool`, `clarification`, `risk_stop` — and from a smaller generative request when it
does happen. On a workload where every step needs generation anyway, expect latency to be
neutral at best.

## OpenClaw is routed, not governed

The OpenClaw adapter decides **which agent** gets a message and whether a human should see it
first. It does **not** sit inside OpenClaw's agent loop — that loop is Node, and there is no
Python seam between its model call and its tool execution. Nothing in this package constrains
what an OpenClaw agent does once dispatched.

## Pydantic AI output tools are left alone

`output_type=` tools are how a run returns a typed result rather than a side effect. The
adapter does not filter or force them: removing them would break the agent's contract with its
caller, and governing them would mean deciding whether the run may *return at all*, which is a
different question from the one this package asks.

## Streaming is not governed

`GovernedModel` implements `request`, not `request_stream`. A streaming run falls through to the
wrapped model ungoverned. If you stream, the governor is not in that path.

## What has and has not been verified

| | Status |
|---|---|
| Decision parsing, guards, projection, config, transport, both adapters' logic | **210 hermetic tests, green** |
| The wire protocol shape | Verified against the live API, 2026-09-25 |
| `scripts/verify_live.py` end-to-end | Runs on demand; **requires your key**, not part of CI |
| Pydantic AI adapter against a real provider | **Not verified in CI** — written against pydantic-ai 2.54 `WrapperModel`; please report breakage |
| Hermes adapter against a live Hermes checkout | **Not verified in CI** — the seam is six lines; verify in your own fork |
| OpenClaw dispatch against a live gateway | **Not verified in CI** — transport properties are tested, a real round trip is not |

The bottom four rows are where to be sceptical. Issues welcome.
