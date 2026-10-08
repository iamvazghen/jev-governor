# jev-governor

**A calibrated System One decision layer for agents you already have.**

[![CI](https://github.com/iamvazghen/jev-governor/actions/workflows/ci.yml/badge.svg)](https://github.com/iamvazghen/jev-governor/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/jev-governor.svg)](https://pypi.org/project/jev-governor/)
[![Python](https://img.shields.io/pypi/pyversions/jev-governor.svg)](https://pypi.org/project/jev-governor/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

An ordinary agent loop asks one generative model to do two unrelated jobs on every step:
**decide what to do next**, and **write the thing**. The first job is a classification over a
fixed set of options. Autoregressive generation is a strange and expensive way to pick one of
eight strings — you pay for reasoning tokens, you re-send every tool description so the model
can choose between them, and you wait seconds for an answer that was never prose.

`jev-governor` splits the loop. A non-autoregressive judgment model (**Jev**, TypeSafe's System
One model) decides; the generative model is then **constrained to that decision and checked
against it**. It installs into agents you already run — Hermes, Pydantic AI, OpenClaw, or any
OpenAI-compatible loop — without changing their tools, memory, sandboxes, or prompts.

```text
             ┌──────────────────────────── one decision, ~70–300 ms ──┐
             │                                                        │
  state ──►  │  action   : choice over YOUR tools + finish/clarify/delegate
             │  goal_met : noul  — is the substantive work already done?
             │  risk     : choice over low/medium/high  (triage, not permission)
             │                                                        │
             └───────────────────────────────┬────────────────────────┘
                                             │
   ┌─────────────────────────────────────────┼─────────────────────────────────────┐
   ▼                     ▼                   ▼                 ▼                   ▼
 static_tool        tool_arguments     final_response      risk_stop           delegated
 pre-approved       model writes ONLY  model writes ONLY   ask a human       unconstrained
 payload, zero      the arguments      the prose           first             generative step
 generative cost    (forced tool)      (tools forbidden)   (no model call)   (as before)
```

---

## Why this is not just a router

Plenty of projects put a cheap classifier in front of an LLM. Three things here are different,
and they are the reason this is a decision *layer* rather than a suggestion.

**1. The decision is enforced, not advised.** When Jev picks a tool, the host constrains the
provider to exactly that function (`tool_choice` + `parallel_tool_calls=False`) and then
`Governor` compares what came back against what was asked for. If a provider ignores the
constraint — and they do, under load, with older models, through proxies — the turn **raises
instead of executing a tool nobody selected.** Without this check, "System One decides" is a
comment in a README. With it, it is a property.

**2. Completion is judged independently of the decision to stop.** `action` and `goal_met` are
separate questions in the same request, so they cannot see each other's answers. A model that
has just chosen `finish` is a poor witness to whether finishing is justified. When the two
disagree, the stricter completion judgment wins and the turn keeps working. This is the single
most common real failure of agent loops — confidently reporting success on work that never
happened — and it is cheap to gate once you ask the question separately.

**3. Incoherent answers are loud instead of silent.** A typed answer is not a trustworthy
answer. Every response is checked: the distribution must cover exactly the offered options and
sum to one, the returned choice must be its own argmax, and reported confidence is **clamped to
the winning option's actual probability mass**. A `JevError` has exactly one meaning — *no
decision was executed* — so it is always safe to retry and never raised after an effect.

---

## Install

```bash
pip install jev-governor                      # core: one dependency (httpx)
pip install "jev-governor[pydantic-ai]"       # the model-wrapper adapter
pip install "jev-governor[static]"            # validated zero-cost static tool payloads
```

You need a TypeSafe API key in `TYPESAFE_API_KEY`. Get one at
[typesafe.ai](https://typesafe.ai). The test suite does **not** need it — see
[Verification](#verification).

### Pydantic AI — wrap the model, change nothing else

```python
from pydantic_ai import Agent
from jev_governor.adapters.pydantic_ai import GovernedModel

agent = Agent(GovernedModel("openai:gpt-4o"), tools=[check_port, read_log])
print(agent.run_sync("Is port 8080 listening? If not, check uptime.").output)
```

Your tools, prompts, and run calls are untouched. Pydantic AI has no `tool_choice`, so the
constraint is applied by re-issuing the request with `function_tools` filtered to the chosen
tool and `allow_text_output=False` — one legal move, using only documented fields.

### Hermes Agent — one seam, six lines

Hermes funnels every provider call through `agent/turn_api_call.py`, so the install is a seam
rather than a rewrite. Full copy-paste in **[docs/install-hermes.md](docs/install-hermes.md)**;
the shape is:

```python
# agent/turn_api_call.py
def _perform_api_call(next_api_kwargs):
    governor = getattr(agent, "_jev_governor", None)
    if governor is not None:
        return governor.call(agent, next_api_kwargs, _generate_api_call)
    return _generate_api_call(next_api_kwargs)
```

Enable per profile, so switching it off is one line and not an uninstall:

```yaml
jev:
  enabled: true
  confidence_threshold: 0.8
  on_error: delegate
```

Tools, sandboxes, session persistence, memory, skills, and compression stay upstream code.

### OpenClaw — route the fleet

OpenClaw runs its own loop in Node, so there is no Python seam inside it. **This adapter does
not claim one.** What it governs is the decision OpenClaw genuinely leaves open — *which agent
gets this, and should a human see it first*:

```python
from jev_governor import JevClient
from jev_governor.adapters.openclaw import AgentSpec, OpenClawRouter

router = OpenClawRouter(JevClient(key), [
    AgentSpec("finance",  "Invoices, bookkeeping, tax deadlines, portfolio questions."),
    AgentSpec("personal", "Household bills, appointments, personal errands and admin."),
    AgentSpec("security", "Breach alerts, credential hygiene, suspicious access."),
])

route = router.route(inbound_message)
print(router.dispatch(route, inbound_message) if route.routable else f"Hold: {route.reason}")
```

Four transport properties, each a deliberate refusal: commands are built as **`argv`, never a
shell string**; the message crosses in a **file** so nothing re-parses `&` or a quote;
**`--deliver` is never passed**, so a turn produces a reply and *sends nothing*; and the child
environment is an **allowlist built from nothing**.

### Anything OpenAI-compatible

```python
from jev_governor import Governor, JevClient
from jev_governor.adapters.openai_chat import OpenAIChatHost

governor = Governor(JevClient(key))
response = governor.step(OpenAIChatHost(request, lambda r: client.chat.completions.create(**r)))
```

---

## What it costs, and what it saves

Honest accounting, because the interesting number is not the one in the benchmark table.

| | Standard loop | With `jev-governor` |
|---|---|---|
| Routing decision | Full generative pass; tokens spent to emit an enum | One System One call, no tokens generated |
| Tool descriptions | Re-sent every step so the model can choose | Not sent for the decision; Jev scores criteria |
| Deciding to stop | Same pass that chose to stop | Separate, independently gated judgment |
| Risk triage | Prompt instructions, if any | A typed answer with a calibrated probability |
| Provider ignoring tool choice | Executes whatever it called | Raises; nothing executes |
| Why did it do that | Read the transcript | One event: path, probabilities, resolved model |

**Jev costs real money per call, and the governor adds one call per step.** It pays for itself
when the generative model you are routing with is expensive and the routing decision is easy —
which is the common case, and is why `delegated` is a *designed* path rather than a failure.
It does not pay for itself if your agent is already running a cheap local model, and
`docs/limits.md` says so plainly.

Numbers from the vendor (~70–300 ms, `$0.042`/M input tokens) are **not reproduced here as
findings.** Measure on your own traffic: `Governor.events` records `decision_ms`, the resolved
model id, and token usage for exactly this purpose.

---

## Configuration

```python
from jev_governor import Governor, GovernorConfig, JevClient

governor = Governor(JevClient(key), GovernorConfig(
    confidence_threshold=0.80,   # below this, delegate instead of routing
    completion_threshold=0.95,   # below this, refuse to stop
    state_chars=24_000,          # bound on the per-step state projection
    timeout=5.0,
    on_error="delegate",         # or "stop"
    goal_hint="You operate a home lab and a small business.",
    static_tools={"docker_ps": {}},   # zero-cost pre-approved payloads
))
```

The two thresholds are deliberately not one knob. The failure modes are not symmetric: a
needless delegation costs a few cents, while a turn that stops early hands the user a confident
answer about work that never happened. Unknown config keys are an **error**, not a warning — a
typo in `confidence_threshold` that is silently ignored leaves the gate at its default while you
believe it was tightened.

---

## Verification

```bash
uv run pytest                    # hermetic: no network, no API key, no provider
uv run python scripts/verify_live.py   # opt-in, needs TYPESAFE_API_KEY, costs a few cents
```

The suite is offline by design. A test suite that needs a paid credential does not run in CI,
and one that does not run in CI is decoration. Live verification is a separate script you run
deliberately.

---

## Documentation

| | |
|---|---|
| [docs/architecture.md](docs/architecture.md) | The two-tier split, the guard order, and why each guard sits where it does |
| [docs/install-hermes.md](docs/install-hermes.md) | Copy-paste Hermes integration and how to keep merging upstream |
| [docs/limits.md](docs/limits.md) | What this cannot do, and when not to use it |
| [docs/comparison.md](docs/comparison.md) | Against a standard loop, and against other routing approaches |
| [docs/roadmap.md](docs/roadmap.md) | Possible improvements, with the reasoning for each |
| [docs/adapters.md](docs/adapters.md) | Writing a `Host` for a framework not listed here |

Two runnable examples: [`examples/openai_loop.py`](examples/openai_loop.py) is a complete
governed agent loop in about fifty lines, and
[`examples/fleet_router.py`](examples/fleet_router.py) routes across a real 18-agent roster
covering 83 utilities — adjacent remits included, which is where a router either earns its keep
or quietly guesses.

---

## Security

Credentials are read from the environment or the host framework's own secret scope, never from
this package's configuration — a credential should have exactly one home. Exception messages
carry status codes but never response bodies, transcript content, or keys. Tool output and
fetched pages are declared to the model as **evidence, never instructions**, because a decision
layer that reads routing commands out of a scraped page subverts the one component whose job is
to be unsubvertible.

Report vulnerabilities per [SECURITY.md](SECURITY.md). Please do not open a public issue for
one.

---

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) covers the dev loop and the review bar. The short version:
a new guard needs a test that fails without it, and docstrings explain **why** a check exists,
not what the line below does.

## License

[MIT](LICENSE). See [NOTICE](NOTICE) for attribution.

`jev-governor` is an independent community project. It is not affiliated with, endorsed by, or
supported by TypeSafe AI or Nous Research. **Jev** and **TypeSafe** are names of a separately
hosted service; this repository distributes no model weights, API keys, or usage credits, and
use of the service is subject to [TypeSafe's terms](https://docs.typesafe.ai/legal).
