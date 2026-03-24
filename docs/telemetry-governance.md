# AstraWeave Telemetry and Privacy Governance (v1)

## 1. Policy Defaults

Default posture:

- Telemetry collection is local-only by default.
- Remote export is disabled by default.
- Diagnostic bundles are opt-in and operator-triggered.
- Prompt/output payload content is never captured.

## 2. Data Classification

Telemetry classes:

- Operational health: startup/shutdown, crash, service status.
- Performance events: transfer latency, migration volume, fallback timing.
- Policy events: tier decisions, fallback reason codes, pressure transitions.
- Security events: auth denies, rate-limit triggers, malformed payload rejects.

Sensitive classes:

- User text payloads, model outputs, file contents, credentials, and secrets are forbidden.

## 3. Schema Inventory

Required event schemas:

- `residency_change`: session id, tensor class, from tier/state, to tier/state, reason code, timestamp.
- `transfer_event`: session id, bytes moved, direction, latency ms, mode, reason code.
- `fallback_event`: session id, ladder step, trigger, stabilization mode flag, reason code.
- `pressure_snapshot`: session id, vram budget, vram used, host pinned used, pressure level.
- `security_event`: caller id hash, deny reason, endpoint, rate-limit bucket.

Required field standards:

- Use correlation id on every event.
- Use stable reason codes from contract.
- Hash caller/session identifiers in persisted logs unless debug mode is explicitly enabled.

## 4. Redaction Rules

Mandatory redaction:

- Remove path segments that contain user names unless explicitly allowlisted.
- Replace raw identifiers with irreversible hashes in persisted events.
- Strip payload-like strings from exception traces before persistence.

Forbidden logging:

- Raw prompts.
- Raw model completion text.
- Secrets from environment variables or config files.

## 5. Retention Matrix

| Data Class | Storage Location | Default Retention | Max Retention |
| --- | --- | --- | --- |
| High-volume event stream | Local ring buffer | 24 hours | 72 hours |
| Aggregated metrics | Local metrics store | 30 days | 90 days |
| Crash metadata | Local diagnostics folder | 14 days | 30 days |
| Diagnostic bundles | Operator-selected export path | 7 days | 30 days |

Retention enforcement:

- Automatic cleanup runs daily.
- Operator cannot configure retention above max policy bounds.

## 6. Export Controls

Export requirements:

- Disabled by default.
- Requires explicit operator action per export request.
- Export bundle must include manifest of included schema types.
- Bundle creation must run redaction pass before write completion.

Auditability:

- Every export action records who triggered it, when, and which schemas were included.

## 7. Compliance and Release Gating

Release requirements:

- Privacy test suite passes (local-only default, opt-in export, redaction correctness).
- Schema inventory and retention matrix are published and versioned.
- No critical privacy findings remain unresolved at release candidate cut.

