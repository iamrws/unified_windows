# AstraWeave Threat Model (v1)

## 1. Scope

This threat model covers:

- Service IPC boundary.
- Session isolation and memory residency orchestration.
- Cross-device migration pathways (RAM/VRAM/CPU offload).
- Telemetry collection, retention, and export.

## 2. Assets

Primary assets:

- Session model state and tensor metadata.
- Session residency and pressure metadata.
- Runtime policy profile and fallback state.
- Diagnostic and telemetry records.
- Service availability and integrity.

## 3. Actors

Actors considered:

- Trusted same-user client process.
- Untrusted local process on same machine.
- Administrator/operator with local policy control.
- Malformed or compromised adapter/runtime caller.

## 4. Trust Boundaries and Entry Points

Trust boundaries:

- Client -> service IPC endpoint.
- Service control plane -> memory orchestration engine.
- Service telemetry pipeline -> local persistence/export workflow.

Entry points:

- API methods: `CreateSession`, `LoadModel`, `RegisterTensor`, `SetTierHint`, `PrefetchPlan`, `RunStep`, `GetResidency`, `GetPressure`, `SetPolicy`, `CloseSession`.
- Diagnostic bundle generation.
- Local policy and configuration loading.

## 5. Threat Inventory

| ID | Threat | Severity | Mitigations | Validation Hook |
| --- | --- | --- | --- | --- |
| T-001 | Unauthorized local caller invokes high-impact APIs | High | Same-user default authz, default-deny allowlist policy, explicit caller requirement, per-connection caller binding, optional transport authkey | Security test: unauthorized caller rejection |
| T-002 | Cross-session data exposure through shared handles or mixed telemetry | High | Session-scoped capabilities, strict session filtering on read APIs, redaction and correlation-id scoping | Security test: session isolation |
| T-003 | Malformed IPC payload causes crash, corruption, or RCE | High | Strong schema validation, bounds checks, size limits, defensive parsing | Fuzzing and malformed payload tests |
| T-004 | Abuse flood degrades service (DoS) | Medium | Per-caller rate limits, queue depth caps, admission control, backoff | Abuse simulation tests |
| T-005 | Telemetry misuse leaks sensitive user data | High | Local-only default, no prompt/output capture, export opt-in, redaction policy | Privacy suite and bundle inspection |
| T-006 | Stale token/session reuse after close | Medium | Idempotent close invalidates session capabilities, strict state checks | API lifecycle tests |
| T-007 | Policy oscillation creates controllable instability path | Medium | Mandatory anti-oscillation cooldown, churn trigger thresholds | Stability tests under churn |

## 6. Severity Policy and Release Blocking

Severity definitions:

- High: may compromise confidentiality, integrity, or availability in default deployment.
- Medium: material reliability or abuse impact, but bounded by service controls.
- Low: limited impact, non-default preconditions, or narrow exploitability.

Release-blocking rule:

- v1 release is blocked while any High-severity threat is unresolved, unmitigated, or lacks passing validation coverage.
- Medium threats require mitigation plan and tracked owner before release candidate.

## 7. Required Mitigation Status for v1

Required before code freeze:

- T-001 through T-003 and T-005 implemented and tested.
- Threat inventory reviewed at each release candidate cut.
- New APIs cannot ship without threat entry update and mitigation mapping.
