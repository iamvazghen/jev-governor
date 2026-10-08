# Architecture

## The problem this solves

A conventional agent loop looks like this:

```text
transcript + every tool schema ──► generative model ──► reasoning tokens ──► a tool name
```

The output of all that work is **one string from a known set**. You paid for a full
autoregressive pass, you re-sent forty tool descriptions so the model could compare them, and
you waited two to seven seconds. The expensive, creative, open-ended machine was used as a
switch statement.

That is not just wasteful — it is *unobservable*. When the agent does something strange, the
only artefact is a transcript. There is no number anywhere saying how close the call was.

## The split

```text
                           ┌─────────── System One ───────────┐
  bounded state  ────────► │  action   : choice over tools     │
  (fresh projection,       │  goal_met : noul                  │  one call, 3 answers
   never the raw           │  risk     : choice low/med/high   │  no tokens generated
   transcript)             └───────────────┬──────────────────┘
                                           │
                        ┌──────────────────┴────────────────────┐
                        ▼                                       ▼
            decision is actionable                   decision is not actionable
         ┌──────────────┴──────────────┐          ┌─────────────┴─────────────┐
         ▼              ▼              ▼          ▼                           ▼
   static payload   force tool    forbid tools   delegate                  stop/ask
   (no model call)  + verify      + verify       (unconstrained,           (no model call)
                                                  exactly as before)
```

System One decides **what**. The generative model supplies **how** — the argument string, the
prose — and only when asked. That boundary is the whole design, and it follows from one fact
about the model: it is non-autoregressive, so it cannot write a bash command, compose a query,
or draft a reply. It can only answer typed questions about state. Trying to make it do more is
the main way this architecture is implemented badly.

## Why the guards are in that order

`Governor._dispatch` runs seven checks. The order is load-bearing.

**1. Interrupted?** A decision takes a few hundred milliseconds, which is long enough for
someone to hit Ctrl-C. Checking after the decision and before the effect costs nothing and is
the last honest moment to notice.

**2. Truncated view and a `finish`?** Delegate. The one judgment that must never be made from a
partial transcript is *"the work is already done"* — because the part that got dropped is
exactly where the unfinished work would be. A tool route from partial evidence is fine; the next
turn sees the result. A *stop* from partial evidence is how an agent confidently reports success
on work it never did.

**3. Confidently high risk?** Stop and ask a human. This sits **before** the confidence gate on
purpose. If it came after, an action judged risky *but with low confidence* would fall through
to the generative model — i.e. the response to "this might be dangerous and I am not sure" would
be to hand it to the component with fewer brakes. Risk is checked first so that uncertainty
about risk never buys an escalation in capability.

**4. Unconfident, or explicitly `delegate`?** Hand it over, unconstrained. **This is the
designed path, not the failure path.** It is supposed to be common. The fast path can afford to
be strict precisely because falling off it is cheap and safe: the agent behaves exactly as it
did before this package was installed.

**5. `clarify`?** Ask the user. No generative call — spending one to phrase a question is the
cost this package exists to avoid.

**6. `finish`?** Only if `goal_met` clears its own, higher threshold. Two thresholds rather than
one because the failure modes are asymmetric: a needless delegation costs a few cents, a
premature stop costs the user's trust.

**7. A tool?** Either a pre-approved static payload (validated against the live schema, and only
when risk is `low`), or the generative model writes the arguments under a hard constraint.

## Enforcement is the load-bearing part

Everything above would be advice if step 7 ended at *asking* the provider nicely.

```python
response = host.generate(force_tool="read_file", directive=...)
called = host.called_tools(response)
if called != ["read_file"]:
    raise JevError("The provider ignored the selected tool; execution blocked")
```

The host sets `tool_choice={"type": "function", "function": {"name": ...}}` and
`parallel_tool_calls=False`. Then the result is **checked**. Providers ignore `tool_choice` —
under load, on older models, through proxies and compatibility shims. Without the check, a
provider that disregarded the constraint would have its own choice executed, and the decision
layer would have been decorative in exactly the cases where it mattered.

The same check runs inverted for `finish`: a tool call after a no-tool decision also raises.

## Distrusting the answer

A typed response is schema-valid, not coherent. Four properties are enforced in
`decision.py`, and each one has a failure behind it:

| Check | What it catches |
|---|---|
| Not a `bool` | `isinstance(True, int)` is true in Python, so a stray boolean reads as certainty |
| Distribution covers exactly the offered options, sums to 1 | The question and the answer have drifted apart |
| The choice is its own argmax | An internally inconsistent answer where neither half can be believed |
| `confidence = min(stated, mass[winner])` | Confidence summarising concentration more optimistically than the mass supports |

None of this makes a judgment *correct*. It makes an incoherent one **loud**, which is the only
guarantee a client library can honestly offer. `confidence` is distribution concentration —
not the probability the action is right, and **never permission to act**.

## The state projection

The governor re-asks on every step, so the size of what it sends is the entire cost model.
`state.py` builds a fresh projection each time and throws it away:

```json
{
  "goal": "the latest user message",
  "instructions": "system messages, concatenated once, in order",
  "recent_history": [ "newest turns that fit, oldest-first" ],
  "truncated": true
}
```

Two properties matter more than the shape.

**The canonical transcript is never touched.** Not truncated, not reordered, not rewritten. It
is what the generative model and the user's own history depend on; a decision layer that edits
it to make its own job cheaper has corrupted the record.

**The bound is enforced on the serialized length**, not the sum of field lengths. JSON escaping
can multiply a string several times over — a transcript full of control characters overruns a
naive character count badly, and the limit would silently stop holding.

And `truncated` is reported rather than hidden, because guard 2 depends on it.

## What it costs

One System One call per step, plus the generative call when one is needed. It pays off when the
model you are routing with is expensive and the routing decision is easy. It does **not** pay
off if you are already routing with a cheap local model — see [limits.md](limits.md).

`Governor.events` records `decision_ms`, the resolved model id (e.g. `jev-1.13.0`, so a
behaviour change is attributable after the fact), token usage, and which of the nine paths the
turn took. Measure on your own traffic; do not trust the table in anyone's README, including
this one.
