# AstraWeave v1 Normative Specification

## 1. Scope and Source of Truth

This document is the normative baseline for AstraWeave v1.

- Primary v1 strategy: `NUMA_dGPU` reliability-first (8 GB VRAM class).
- Secondary optimized path: `CacheCoherentUMA` (must not block v1 release).
- Telemetry default: local-only, export disabled until explicit opt-in.

Normative companion documents:

- `docs/security-architecture.md`
- `docs/threat-model.md`
- `docs/api-contract.md`
- `docs/telemetry-governance.md`
- `docs/operations-readiness.md`
- `docs/compatibility-matrix.md`
- `docs/release-governance.md`
- `docs/test-plan.md`

`problems.md` is treated as a hard gate register. P0 and P1 gates must be closed before code freeze.

## 2. Decision Log (Closed Decisions)

| ID | Decision | Outcome | Owner | Date |
| --- | --- | --- | --- | --- |
| D-001 | Runtime vs framework policy boundary | AstraWeave owns cross-tier placement, residency, migration, fallback, and observability. Frameworks own model graph execution and allocator internals. | Core Runtime Lead | 2026-03-24 |
| D-002 | Tunability level | Stability profile is default with safe knobs only. Throughput profile is opt-in and cannot disable safety rails. | Product + Runtime | 2026-03-24 |
| D-003 | Minimum hardware bar | v1 acceptance target is Windows 11 + 8 GB dGPU + >= 32 GB RAM; design-optimized target remains 128 GB RAM host. | Product | 2026-03-24 |
| D-004 | Telemetry surface default | Local-only telemetry is default. Export and diagnostic bundle creation require explicit operator action. | Security + Product | 2026-03-24 |
| D-005 | Hardware strategy contradiction | dGPU reliability-first is primary release path. `CacheCoherentUMA` is secondary fast path with separate acceptance gates. | Architecture Council | 2026-03-24 |

## 3. Runtime Architecture and Ownership Boundaries

Required service modules:

- Hardware Profiler: detects `CacheCoherentUMA`, `UMA`, `NUMA_dGPU`, and key budget/capability limits.
- Budget Tracker: tracks WDDM budget and pressure snapshots.
- Tier Engine: classifies `HOT`, `WARM`, `COLD` and computes residency plans.
- Transfer Scheduler: executes prefetch, migration, and eviction.
- Fallback Controller: enforces mandatory ladder and anti-oscillation controls.
- Telemetry Pipeline: emits stable reason-coded events under governance policy.

Ownership boundaries:

- AstraWeave service owns memory policy, migration decisions, and fallback orchestration.
- SDKs own request marshalling, retries, cancellation propagation, and error decoding.
- Inference runtime adapters (for example `llama.cpp`) own tensor registration hints and step boundary signals.

## 4. Security, Threat Model, and API Contract

Normative security/trust boundary requirements are in `docs/security-architecture.md`.

Normative threat model and release-blocking risk criteria are in `docs/threat-model.md`.

Normative API lifecycle, concurrency, error taxonomy, and compatibility guarantees are in `docs/api-contract.md`.

Mandatory lifecycle:

`CreateSession -> LoadModel -> RegisterTensor/SetTierHint/PrefetchPlan -> RunStep -> CloseSession`

Out-of-order calls must fail with typed `ApiError` codes.

## 5. Hardware Strategy, Fallback Ladder, and Stability Controls

Capability modes:

- `NUMA_dGPU` (primary v1 release path).
- `CacheCoherentUMA` (secondary optimized path).
- `UMA` (non-coherent, supported with stricter stall controls).
- Unsupported profiles fail fast with actionable diagnostics.

Mandatory fallback ladder order:

1. Reduce KV/context window.
2. Reduce batch.
3. Lower precision.
4. Selective CPU offload.
5. Controlled fail with diagnostics and stable reason code.

Anti-oscillation controls:

- Cooldown window after each ladder step: 30 seconds.
- Minimum residency dwell time before re-promotion: 15 seconds.
- Churn trigger for stability mode: > 1 fallback step change per 30 seconds.

## 6. Telemetry and Privacy Defaults

Normative telemetry policy is in `docs/telemetry-governance.md`.

Hard defaults:

- No remote export by default.
- No prompt/content payload logging.
- Structured events only, with redacted or hashed identifiers where needed.
- Retention and bundle controls are operator-configurable within policy bounds.

## 7. Operational and Release Governance

Operational readiness requirements are in `docs/operations-readiness.md`.

Compatibility/support matrix requirements are in `docs/compatibility-matrix.md`.

Licensing/compliance/signing/distribution requirements are in `docs/release-governance.md`.

## 8. Test and Acceptance Gates

Normative test suites and workload definitions are in `docs/test-plan.md`.

Release gates:

- `P0` gate: zero unresolved high-severity threat-model items.
- Reliability gate: zero hard OOM across 100 repeated 7B/13B runs on target dGPU profile.
- Latency gate: migration-induced `RunStep` p95 stall <= 250 ms.
- Stability gate: fallback oscillation <= 1 ladder step change per 30 seconds after stability mode entry.
- Explainability gate: 100 percent of evictions/fallbacks carry stable reason code and session correlation id.

