# AstraWeave v1 Test Plan and Acceptance Gates

## 1. Workload Definitions

Release-gate workloads:

- `W-7B-CHAT`: 7B chat model, 4k context, sustained decode run.
- `W-13B-CHAT`: 13B chat model, 4k context, sustained decode run.

Execution profile:

- 100 repeated runs per workload on primary `NUMA_dGPU` target profile.
- At least one multi-session contention scenario per run set.

## 2. Unit Tests

Required unit suites:

- Tier classification (`HOT/WARM/COLD`) correctness.
- Eviction ordering under dynamic budget contraction.
- Fallback ladder ordering and stop conditions.
- Anti-oscillation state machine transitions.
- API error-code mapping and state validation.

## 3. Integration Tests

Required integration suites:

- Synthetic RAM/VRAM churn and determinism checks.
- WDDM budget contraction and rehydration recovery.
- Cross-session fairness and responsiveness under contention.
- Capability branch correctness for `NUMA_dGPU` vs `CacheCoherentUMA` vs `UMA`.

## 4. Security and Privacy Tests

Required security suites:

- Unauthorized caller rejection.
- Session isolation validation.
- Malformed payload and bounds fuzzing.
- Rate-limit abuse simulation.

Required privacy suites:

- Local-only telemetry default verification.
- Export opt-in enforcement.
- Redaction correctness.
- Retention expiry behavior.

## 5. Operational Tests

Required operations suites:

- Rollback drill.
- Crash artifact completeness validation.
- Runbook execution drill for at least one `S1` simulation.

## 6. Quantitative Release Gates

Release gates:

- `P0` gate: zero unresolved high-severity items in `docs/threat-model.md`.
- Reliability gate: zero hard OOM events across 100 repeated `W-7B-CHAT` and `W-13B-CHAT` runs on primary target profile.
- Latency gate: migration-induced `RunStep` p95 stall <= 250 ms.
- Stability gate: fallback oscillation <= 1 ladder step change per 30 seconds after stability mode entry.
- Explainability gate: 100 percent of eviction/fallback events include stable reason code and session correlation id.

## 7. Reporting Requirements

Each release-candidate report must include:

- Workload metadata and hardware profile.
- Pass/fail for each gate with measured values.
- Failure artifacts and root-cause notes.
- Explicit release recommendation: block, conditional pass, or pass.

