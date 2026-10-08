# Roadmap

Each item says what it is, **why it would matter**, and what would have to be true to accept it.
Items are not promises. A few are deliberately listed as *probably not*, with the reasoning,
because a roadmap that only contains good ideas is not telling you much.

## Likely next

### 1. An async `Governor`

`AsyncJevClient` exists, but `Governor.step` is synchronous, so the Pydantic AI adapter runs it
on a worker thread. That works and does not block the caller's loop, but it is a thread per
step for what is fundamentally one await.

*Why it matters:* a server handling concurrent agent runs pays a thread each.
*Accept when:* `AsyncGovernor` shares the guard logic with the sync one rather than duplicating
it — two copies of the guard order is exactly the bug this package exists to prevent.

### 2. Streaming support

`request_stream` currently falls through to the wrapped model ungoverned.

*Why it matters:* interactive agents stream, so the governor is absent from the path most users
see.
*The hard part:* the decision has to be made before the stream opens, and `called_tools`
verification needs the stream to finish. Likely shape: decide, constrain, open the stream, and
verify at completion — accepting that a violation is caught after bytes have been sent, which
must be documented rather than glossed over.

### 3. Per-tool risk floors

Today one `confidence_threshold` covers every tool. `rm_rf` and `read_file` deserve different
bars.

*Why it matters:* a single global threshold is either too loose for the dangerous tools or too
tight for the safe ones, and people will set it loose.
*Shape:* `tool_thresholds: {"delete_file": 0.97}`, defaulting to the global value.

### 4. A recorded-decision test harness

Capture real `(state, questions, response)` triples, replay them offline.

*Why it matters:* it is currently impossible to answer "would the new threshold have changed
anything?" without spending money. This turns threshold tuning from guesswork into a
measurement.
*Accept when:* the recording format redacts transcript content by default — a decision corpus
is a transcript corpus, and that is the owner's data.

## Worth doing, less clear how

### 5. Multi-step lookahead

Ask about the next *two* steps in one call and act on both when the pair is confident.

*Why it might matter:* halves decisions on long deterministic chains.
*Why it might not:* the second step is conditioned on a result that does not exist yet, so the
answer is a guess dressed as a decision. Would need evidence that paired confidence is
calibrated before it could be trusted.

### 6. A cache keyed on the projection

Identical state and questions could skip the call.

*Why it might matter:* retry loops and idempotent polling re-ask the same question.
*Why it is risky:* agent state usually differs by a timestamp, so the hit rate may be near
zero while the staleness risk is real. Measure the hit rate on recorded traffic (item 4) before
building it.

### 7. More adapters — LangGraph, CrewAI, Mastra, AutoGen

*Why it matters:* the `Host` protocol is six methods; the work is mostly understanding each
framework's constraint mechanism.
*Accept when:* the adapter can genuinely *force* a tool choice in that framework. An adapter
that can only suggest would quietly reduce this to advice, which is worse than no adapter —
see [adapters.md](adapters.md).

### 8. OpenTelemetry spans

*Why it matters:* `Event` is a list in memory; production wants it in the same trace as the
provider call.
*Keep:* the in-memory list, so the package stays dependency-light and `events` keeps working
with no collector.

## Probably not

### 9. Letting Jev write tool arguments

It cannot — it is non-autoregressive and emits no tokens. Any design that appears to do this is
really a `choice` over pre-enumerated payloads, which is `static_tools` with extra steps.

### 10. Making this a full agent framework

The value here is that it is **additive**. A framework competes with the one you already run;
a decision layer improves it. Growing a tool registry, a memory store and an execution sandbox
would mean maintaining worse versions of things Hermes and Pydantic AI already do well.

### 11. Bundling a provider SDK

One runtime dependency (`httpx`) is a feature. This installs into someone else's agent, and
every dependency it drags in is a version constraint they did not ask for.

### 12. Treating `risk` as authorization

Repeatedly requested in spirit — "can it just approve low-risk actions?" No. A calibrated
probability is not a permission, and the moment it is treated as one, a confident wrong answer
becomes an executed action. Authorization stays with the host.

## Contributing an item

Open an issue with the *failure* you hit, not the feature you want. Several items above exist
because someone described a bad turn precisely enough to see the missing guard in it.
