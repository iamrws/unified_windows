# AstraWeave Security Architecture (v1)

## 1. Trust Boundary

Trust boundaries:

- Boundary A: client process to local AstraWeave service IPC endpoint.
- Boundary B: session state and memory residency metadata between tenants/callers.
- Boundary C: telemetry pipeline to local storage and optional export workflow.

Security model:

- Local-only service boundary.
- Default-deny caller policy.
- Same-user access by default.
- Explicit allowlist required for any cross-user service operation.

## 2. IPC Contract and Caller Identity

IPC requirements:

- Transport: local named pipe preferred on Windows; localhost TCP fallback supported for compatibility/testing.
- Non-localhost network listeners: forbidden in v1.
- IPC payload format: strict schema-validated structured messages.
- Max control-plane payload size: 1 MiB.

Caller authentication:

- IPC server requires explicit caller identity by default (`user_sid`, `pid`) for every request.
- IPC server binds each connection to the first accepted caller identity and rejects caller switching.
- Caller authorization is enforced via `SecurityGuard` policy checks before service dispatch.
- Optional transport authkey is supported via `ASTRAWEAVE_IPC_AUTHKEY`.
- Service records `caller_sid` and `caller_pid` in session metadata.
- Service rejects callers failing identity checks with `AW_ERR_AUTH_DENIED`.

Caller authorization:

- Same-user caller receives default rights.
- Cross-user caller requires explicit local policy allowlist entry.
- Unknown caller defaults to deny.

## 3. Session Isolation

Isolation rules:

- Each session has unique opaque `session_id` and scoped capabilities.
- Session memory handles and residency views are never shared across session ids.
- `GetResidency` and `GetPressure` must return session-scoped views only.
- Session close invalidates all outstanding session capabilities.

Idempotency and safety:

- `CloseSession` is idempotent.
- Access with stale session capability returns `AW_ERR_NOT_FOUND` or `AW_ERR_INVALID_STATE`.

## 4. Abuse Controls

Per-caller controls:

- `CreateSession` rate limit: 6 requests per minute.
- Concurrent sessions per caller: max 8.
- `RunStep` single-flight per session: max 1 active call.

Service-level controls:

- Deterministic typed error responses for malformed requests and denied calls.
- Session-bound authorization checks for every state-mutating API.

## 5. Data Handling Rules

Forbidden telemetry content:

- Raw prompts.
- Raw generated model output.
- File contents from caller context.
- Access tokens, secrets, credentials.

Allowed telemetry content:

- Reason codes, durations, sizes, counters, mode flags, redacted identifiers.

## 6. Security Requirements for Implementation

Mandatory implementation requirements:

- Schema validation and field bounds checks on every IPC request.
- Integer overflow, allocation bounds, and copy length checks in service core.
- Redaction pass before telemetry persistence.
- Stable audit trail for security-relevant denies and failures.

Release-blocking requirements:

- No unresolved high-severity threats in `docs/threat-model.md`.
- Security test suite pass for auth, isolation, malformed input, and abuse controls.
