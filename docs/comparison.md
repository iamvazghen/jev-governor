# Comparison

## Against a standard agent loop

The claims below are **structural** — they follow from what the code does, and the hermetic
suite pins each one. They are not performance measurements; see
[limits.md](limits.md#vendor-numbers-are-not-reproduced-here-as-findings).

| Property | Standard loop | With `jev-governor` |
|---|---|---|
| Who picks the tool | Generative model, in prose it then parses | System One, a typed `choice` over a closed set |
| Tool descriptions per step | Re-sent so the model can compare them | Not sent for the decision |
| Tokens generated to decide | Reasoning tokens, then the call | None |
| Deciding to stop | The same pass that chose to stop | A separate `noul`, gated at its own threshold |
| Evidence the work is done | The model's assertion | A probability, held against `completion_threshold` |
| Decision on a truncated view | Indistinguishable from a full one | `finish` is refused; `truncated` is recorded |
| Risk assessment | Prompt text, if any | A typed `choice` with a calibrated probability |
| Provider ignores `tool_choice` | Its own choice executes | `JevError`; **nothing executes** |
| Zero-cost routine calls | Not possible | `static_tools`, schema-validated |
| Post-hoc explanation | Read the transcript | One `Event`: path, probabilities, model, ms |
| Failure of the decision layer | n/a | `on_error`: degrade to the old behaviour, or stop |

The rows that matter most are the last four. The first few are efficiency; those are
*observability and enforcement*, which a cheaper model alone does not give you.

## Against other ways of doing this

### A small LLM as the router

The common approach: route with GPT-4o-mini or a local 8B, execute with something bigger.

- **Still autoregressive.** You still generate tokens to emit an enum, and you still have to
  parse prose into a decision.
- **No calibration.** You get a tool name, not a distribution. There is no number to threshold
  on, so "the model was unsure" is not a state your code can see — which means no
  `delegated` path, no `completion_held`, no meaningful risk gate.
- **Cheaper per call.** Genuinely. If your routing is easy and your budget is the binding
  constraint, this is a reasonable choice and this package is overkill.

### Hand-written rules or a classifier you train

- **Fastest and cheapest at runtime**, with no network dependency.
- **You own the maintenance.** Every new tool means new rules or new labels, and routing
  intent drifts as the agent's remit grows.
- **Worth it for a stable, narrow toolset.** If your five tools have not changed in a year,
  write the rules. The case for a judgment model is a toolset and a user base that move.

### Framework-native structured output

Pydantic AI, Instructor and friends can make a generative model return a typed object, which
removes the parsing problem.

- **Solves a different problem.** You get a well-shaped answer; you still paid a full
  generative pass to produce it, and the confidence it reports is self-assessed prose, not a
  distribution.
- **Composes fine with this.** Nothing stops you using structured output for the *arguments*
  while System One picks the tool.

### A guardrail or policy layer

- **Guardrails usually sit at the edges** — input filtering, output checking.
- **This sits in the control flow.** The risk gate fires *before* a step runs, and the
  `finish` gate fires before an answer is written. Different position, different failure
  catches. They are complementary, not alternatives.

## When to use this

Good fit:

- An expensive frontier model doing many cheap routing steps.
- A toolset that changes, so hand-written rules rot.
- Long-running or unattended work, where a premature `finish` or an unreviewed risky action is
  expensive.
- You need to be able to answer *why did it do that* with a number.

Poor fit:

- Mostly-writing workloads, where routing was never the cost.
- An already-cheap local model doing the routing.
- A stable five-tool agent — write the rules.
- Hard offline requirements, or a refusal to add a network dependency to control flow.

## Honest summary

The efficiency argument is real but secondary and workload-dependent. The durable argument is
that **an agent's control flow becomes typed, thresholded, and inspectable** — the decision is
a value your code can gate on rather than a sentence it has to trust.
