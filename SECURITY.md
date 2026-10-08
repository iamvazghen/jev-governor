# Security Policy

## Reporting a vulnerability

Please report privately via [GitHub Security Advisories][advisories] — "Report a vulnerability"
on the Security tab. **Do not open a public issue.**

[advisories]: https://github.com/iamvazghen/jev-governor/security/advisories/new

Include what you can: affected version, a minimal reproduction, and the impact you believe it
has. Expect an acknowledgement within a few days; this is a community project without a paid
on-call rotation, so please allow reasonable time before disclosing.

## Scope

In scope — this repository's own behaviour:

- Credential leakage through exception messages, logs, `Event` records, or subprocess
  environments.
- Any path where the enforcement check can be bypassed, so a tool the decision layer did not
  select is executed.
- Prompt injection that causes the decision layer to take routing instructions from tool output
  or fetched content.
- Argument or command injection through the OpenClaw adapter's subprocess path.
- A `static_tools` payload reaching execution without schema validation.

Out of scope:

- Vulnerabilities in TypeSafe's hosted service — report those to TypeSafe.
- Vulnerabilities in Hermes, Pydantic AI, OpenClaw, or any dependency — report upstream.
- **A confidently wrong judgment.** The model can be wrong; that is documented, not a
  vulnerability. See [docs/limits.md](docs/limits.md).
- Treating `risk: low` as authorization. It is explicitly not, in the code and the docs.

## Properties this package tries to hold

Each has a test. If you can break one, that is a report.

**Credentials have exactly one home.** The key is read from the environment or the host
framework's own secret scope — never from this package's config. `JevClient.close()` drops the
reference.

**Exceptions carry status codes, never bodies.** A response body may contain the state the
agent was shown. `JevError` messages never include remote content, transcript fragments, or
keys.

**The decision is enforced, not advised.** A provider that ignores the tool constraint causes a
`JevError` before execution. The inverse holds too: a tool call after a `finish` decision is
blocked.

**Tool output is evidence, never instructions.** Every question declares this. A decision layer
that could be redirected by a scraped page would subvert the one component whose job is to be
unsubvertible.

**`JevError` means nothing ran.** It is raised before any effect, never after one.

**No shell, ever.** The OpenClaw adapter builds `argv` lists and refuses an interpreter
(`bash`, `cmd`, `powershell`, `ssh`, …) as its binary. The message crosses in a file, so
nothing re-parses metacharacters at either end.

**Nothing is sent.** The OpenClaw adapter never passes `--deliver`, a channel, or a recipient.
A test asserts the string literal does not appear in the source at all — not even commented
out, where one edit would arm it.

**The child environment is an allowlist built from nothing.** Only the two gateway variables,
plus what a process needs to run. An agent's credentials are not inherited.

**Static payloads are validated against the live schema**, with remote `$ref` refused. Without
`jsonschema` installed, static execution refuses rather than proceeding unvalidated.

## Supported versions

Pre-1.0: only the latest release is supported. Fixes land in a new patch release.

## Your own responsibilities

- Keep `TYPESAFE_API_KEY` out of source control and out of client-side code.
- Your agent's tool authorization still has to be correct on its own. This package narrows what
  gets attempted; it does not grant or check permissions.
- Choose `on_error` deliberately. `"delegate"` keeps you serving but makes the risk gate
  best-effort; `"stop"` keeps the gate but fails closed.
- `Event.usage` and `Event.model` are safe to log. The state projection is **not** — it
  contains transcript content.
