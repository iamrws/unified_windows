# AstraWeave v1 Execution Plan (dGPU-First, Local-Only Telemetry)

## Summary
- This plan integrates the latest repo analysis: the workspace is currently docs-only, and `problems.md` defines 10 release-blocking gaps.
- v1 is locked to a **dGPU reliability-first primary path** (8 GB VRAM class), with **`CacheCoherentUMA` as a secondary optimized path**.
- Primary objective: move from concept docs to a build-ready, release-gated spec and implementation sequence that closes all `P0` and `P1` gaps before code freeze.

## Implementation Changes
- Create one normative spec baseline in [plan.md](/c:/Users/grzeg/Documents/Jito/testmarch24/unified_windows/plan.md) and treat [problems.md](/c:/Users/grzeg/Documents/Jito/testmarch24/unified_windows/problems.md) as hard release gates.
- Add a decision log in [plan.md](/c:/Users/grzeg/Documents/Jito/testmarch24/unified_windows/plan.md) that explicitly closes: policy boundary, tunability level, minimum hardware bar, telemetry surface.
- Define service trust boundary and IPC contract: local-only IPC, same-user default access, default-deny for unknown callers, per-session isolation, per-client rate limiting.
- Define a formal threat model: assets, actors, entrypoints, trust boundaries, abuse cases, mitigations, and release-blocking severity criteria.
- Lock API lifecycle semantics: `CreateSession -> LoadModel -> RegisterTensor/SetTierHint/PrefetchPlan -> RunStep -> CloseSession`; reject out-of-order calls with typed errors.
- Lock concurrency semantics: session-level serialized `RunStep`, cross-session fairness scheduler, cancellation token for long operations, idempotent session close.
- Lock error taxonomy and compatibility policy: stable error codes, semantic versioning for SDK/service API, backward-compatible additions only in minor releases.
- Keep runtime modules as planned (Profiler, Budget Tracker, Tier Engine, Scheduler, Fallback, Telemetry) and define ownership boundaries between them.
- Resolve hardware strategy contradiction: optimize default policies for `NUMA_dGPU`; implement `CacheCoherentUMA` fast-path behind capability detection with separate acceptance gates.
- Define fallback ladder as mandatory behavior: KV/context reduction -> batch reduction -> precision reduction -> selective CPU offload -> controlled fail with diagnostics.
- Define anti-oscillation controls: cooldown windows, minimum residency dwell time, churn-trigger threshold for stability mode.
- Define telemetry governance: local-only by default, explicit export opt-in, schema inventory, redaction rules, retention limits, diagnostic bundle controls.
- Add operational readiness requirements: incident runbook, rollback triggers, crash artifact standard, ownership matrix, and failure-drill cadence.
- Add compatibility matrix requirements: Windows version, WDDM level, driver branch, GPU class, degraded-mode expectations, and pass/fail criteria.
- Add release governance requirements: SBOM, dependency pinning policy, signing requirements, attribution/compliance checklist, and distribution channel policy.

## Public APIs / Interfaces
- Service API remains: `CreateSession`, `LoadModel`, `RegisterTensor`, `SetTierHint`, `PrefetchPlan`, `RunStep`, `GetResidency`, `GetPressure`, `SetPolicy`.
- Add explicit interface guarantees for each API: preconditions, postconditions, timeout behavior, cancellation behavior, and error codes.
- Add core types: `MemoryTier`, `ResidencyState`, `PolicyProfile`, `PressureSnapshot`, `TransferEvent`, `FallbackEvent`, plus `ApiError` and `SessionState`.
- Python SDK mirrors service API 1:1 and must preserve error-code parity with service responses.

## Test Plan
- Unit tests: tier classification, eviction ordering under budget contraction, fallback ladder ordering, anti-oscillation state transitions, error-code mapping.
- Integration tests: synthetic RAM/VRAM churn, WDDM budget contraction/recovery, multi-session contention fairness, UMA-vs-dGPU capability branch correctness.
- Security tests: unauthorized caller rejection, malformed payload fuzzing, session isolation checks, rate-limit abuse simulation.
- Privacy tests: local-only default verification, export opt-in enforcement, redaction correctness, retention expiry behavior.
- Operations tests: rollback drill, crash artifact completeness, runbook execution drill.
- Quantitative release gates:
- `P0` gate: zero high-severity unresolved threat-model items.
- Reliability gate: zero hard OOM events across 100 repeated 7B/13B scenario runs on target 8 GB dGPU profile.
- Latency gate: `RunStep` p95 stall due to migration <= 250 ms on defined benchmark workload.
- Stability gate: fallback oscillation <= 1 ladder step change per 30 seconds after entering stability mode.
- Explainability gate: 100% of evictions/fallbacks emitted with stable reason code and session correlation id.

## Assumptions and Defaults
- Primary v1 acceptance target is 8 GB dGPU + high RAM host; `CacheCoherentUMA` is secondary and must not block v1 release.
- Telemetry is local-only by default; no remote export unless explicitly enabled by operator/user.
- v1 remains reliability-first, not peak-throughput-first.
- Service + SDK deployment model is retained; CLI-only mode is out of scope for v1.
- Apple-equivalent hardware coherency/SLC behavior is not a v1 goal; AstraWeave targets closest practical Windows behavior via policy and residency orchestration.
