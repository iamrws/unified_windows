# Problems: AstraWeave Audit Gap Register

## Scope and Sources

This document tracks release-blocking gaps and their resolution artifacts across:

- `plan.md`
- `docs/security-architecture.md`
- `docs/threat-model.md`
- `docs/api-contract.md`
- `docs/telemetry-governance.md`
- `docs/operations-readiness.md`
- `docs/compatibility-matrix.md`
- `docs/release-governance.md`
- `docs/test-plan.md`
- `thoughts_1.md`
- `apple-unified-memory-architecture.md`

Audit intent: keep P0/P1 as hard release gates and make evidence requirements explicit.

## Severity Legend

- `P0`: critical blocker for secure/reliable release.
- `P1`: high-priority blocker for delivery confidence and product fit.
- `P2`: important completeness/governance blocker for scale-up.

## Hardware Reality Reset (March 24, 2026)

This addendum reopens hardware execution gates after validating that most existing
evidence is contract/simulation coverage, not end-to-end GPU orchestration.

### Finding 11

- ID: `P0-HW-11`
- Problem: Service orchestration path does not execute real tensor allocation/migration in VRAM.
- Status: `Open`.
- Resolution artifact: `scripts/cuda_transfer_poc.py` and follow-on service integration work.
- Gate evidence required: repeatable host->device->host byte round-trip on target GPU with integrity verification artifacts.

### Finding 12

- ID: `P1-HW-12`
- Problem: Hardware telemetry and orchestration are not yet integrated in the service lifecycle path (`CreateSession`...`RunStep`).
- Status: `Open`.
- Resolution artifact: planned runtime integration in `astrawave/service.py` and related module boundaries.
- Gate evidence required: service-triggered operation showing observable NVML memory deltas and typed failure behavior when GPU resources are unavailable.

### Finding 13

- ID: `P1-HW-13`
- Problem: Release narrative can be interpreted as fully release-ready despite simulation-bound runtime behavior.
- Status: `Mitigated in part` (README now documents real vs simulated boundaries), still open at report/governance level.
- Resolution artifact: README boundary section plus pending report/register alignment updates.
- Gate evidence required: all release-facing summaries explicitly distinguish simulation pass evidence from hardware execution pass evidence.

## Findings and Resolution Artifacts

### Finding 1

- ID: `P0-01`
- Problem: Service security architecture was missing.
- Status: `Validated in implementation, tests, and RC gate evidence`.
- Resolution artifact: `docs/security-architecture.md`
- Gate evidence required: auth/isolation/abuse test pass evidence.

### Finding 2

- ID: `P0-02`
- Problem: No formal threat model existed.
- Status: `Validated` (high-severity mitigation closure sign-off published).
- Resolution artifact: `docs/threat-model.md`
- Gate evidence required: zero unresolved high-severity threats.

### Finding 3

- ID: `P0-03`
- Problem: Telemetry/privacy governance was undefined.
- Status: `Validated in implementation and tests`.
- Resolution artifact: `docs/telemetry-governance.md`
- Gate evidence required: local-only default and redaction/export tests passing.

### Finding 4

- ID: `P0-04`
- Problem: Public API contract was under-specified.
- Status: `Validated` (lifecycle, error taxonomy, and implemented v1 concurrency semantics covered).
- Resolution artifact: `docs/api-contract.md`
- Gate evidence required: state-machine, timeout, and error-code conformance tests.

### Finding 5

- ID: `P1-05`
- Problem: Hardware strategy was inconsistent.
- Status: `Validated` (primary RC workloads plus capability-branch matrix evidence).
- Resolution artifact: `plan.md` Section 1 and Section 5.
- Gate evidence required: primary `NUMA_dGPU` acceptance pass plus secondary `CacheCoherentUMA` branch validation.

### Finding 6

- ID: `P1-06`
- Problem: Open decisions were not closed.
- Status: `Validated against current implementation behavior`.
- Resolution artifact: `plan.md` Section 2 (Decision Log).
- Gate evidence required: implementation docs and behavior align with decisions D-001 through D-005.

### Finding 7

- ID: `P1-07`
- Problem: Acceptance criteria were not measurable.
- Status: `Validated` (quantitative RC workload report attached).
- Resolution artifact: `docs/test-plan.md` Section 6.
- Gate evidence required: measured report for all quantitative thresholds.

### Finding 8

- ID: `P1-08`
- Problem: Operational readiness model was missing.
- Status: `Validated` (auth abuse, budget contraction/recovery, and rollback drills completed).
- Resolution artifact: `docs/operations-readiness.md`
- Gate evidence required: completed drills and rollback validation evidence.

### Finding 9

- ID: `P2-09`
- Problem: Compatibility matrix was missing.
- Status: `Validated` (profiles P-A through P-D covered by repeatable test evidence).
- Resolution artifact: `docs/compatibility-matrix.md`
- Gate evidence required: pass/fail results for profiles P-A through P-D.

### Finding 10

- ID: `P2-10`
- Problem: Licensing/compliance/distribution governance was missing.
- Status: `Validated` (SBOM, attribution, checksums, and signing-verification artifact set published).
- Resolution artifact: `docs/release-governance.md`
- Gate evidence required: SBOM, signing verification, attribution bundle.

## Cross-Document Contradictions

Status update:

- v1 strategy contradiction is resolved at spec level:
  - Primary path is `NUMA_dGPU` reliability-first.
  - Secondary path is `CacheCoherentUMA`.
- Reliability intent is now tied to explicit operational, security, and quantitative gates.

## Missing Decisions Blocking Build/Release

Decision status:

- No unresolved high-impact product decisions remain in the normative spec baseline.
- Prior contract/governance blockers are closed.
- Hardware execution blockers `P0-HW-11`, `P1-HW-12`, and `P1-HW-13` are open and release-blocking for production claims.

## Latest Validation Evidence (March 23-24, 2026)

Execution and evidence artifacts:

- Unit + contract suite:
  - Command: `python -m unittest discover -s tests -v`
  - Result: `56/56` tests passed.
  - Evidence: `reports/runlogs/unittest_rc_2026-03-24.txt`
- RC workload validation (`W-7B-CHAT`, `100` iterations):
  - Command: `scripts/soak_test.ps1 -SkipUnitTests -Iterations 100 -WorkloadId W-7B-CHAT -ModelName W-7B-CHAT -TensorBytes 1048576`
  - Result: `100/100` pass, `run_step_p95_ms=181.811`.
  - Evidence: `reports/runlogs/summary_rc_w7b_chat_2026-03-24.json`
- RC workload validation (`W-13B-CHAT`, `100` iterations):
  - Command: `scripts/soak_test.ps1 -SkipUnitTests -Iterations 100 -WorkloadId W-13B-CHAT -ModelName W-13B-CHAT -TensorBytes 2097152`
  - Result: `100/100` pass, `run_step_p95_ms=183.845`.
  - Evidence: `reports/runlogs/summary_rc_w13b_chat_2026-03-24.json`
- Abuse-control validation (single caller PID):
  - Command: `scripts/soak_test.ps1 -SkipUnitTests -Iterations 50 -RotateCallerPid:$false`
  - Result: `5` admitted, `45` denied with `AW_ERR_RATE_LIMITED` (expected control behavior).
  - Evidence: `reports/runlogs/summary_2026-03-24_001253.json`
- Threat-model closure sign-off:
  - Evidence: `reports/release_gate/threat_signoff_2026-03-24.json`
- Compatibility matrix evidence (`P-A` through `P-D`):
  - Evidence: `reports/release_gate/compatibility_matrix_2026-03-24.json`
- RC quantitative gate report:
  - Evidence: `reports/release_gate/rc_workload_report_2026-03-24.json`
- Operations drill evidence:
  - Evidence: `reports/release_gate/operations_drill_2026-03-24.json`
- Compliance artifact set (SBOM/attribution/checksums/signing verification):
  - Evidence: `reports/release_artifacts/compliance_manifest_2026-03-24.json`
- Consolidated readiness report:
  - Evidence: `reports/release_gate/release_gate_readiness_2026-03-24.json`
- Hardware probe evidence (real `nvidia-smi` + NVML reads):
  - Command: `python -m astrawave.cli hardware-probe`
  - Evidence: `reports/runlogs/hardware_probe_2026-03-24_131123.json`
- CUDA transfer PoC evidence (real host->device->host round-trip):
  - Command: `python scripts/cuda_transfer_poc.py --bytes 1048576`
  - Evidence: `reports/runlogs/cuda_transfer_poc_2026-03-24_131123.json`

## Release Readiness Checklist

Documentation completeness:

- [x] `P0-01` Security architecture document published and linked from `plan.md`.
- [x] `P0-02` Threat model published with severity and closure rules.
- [x] `P0-03` Telemetry governance policy published (schema, retention, privacy defaults).
- [x] `P0-04` API contract published (lifecycle, errors, concurrency, versioning).
- [x] `P1-05` Hardware strategy decision recorded and normalized.
- [x] `P1-06` Open questions closed in decision log.
- [x] `P1-07` Quantitative acceptance thresholds defined.
- [x] `P1-08` Operations readiness requirements published.
- [x] `P2-09` Compatibility/support matrix published.
- [x] `P2-10` Licensing/compliance/signing/distribution governance published.

Validation completeness (release blocking):

- [x] `P0-01` Security controls validated by auth/isolation/abuse test results.
- [x] `P0-02` High-severity threats mitigated or risk-accepted with sign-off.
- [x] `P0-03` Privacy suite validates local-only defaults and redaction/export controls.
- [x] `P0-04` API conformance tests verify lifecycle, error taxonomy, and implemented v1 concurrency rules.
- [x] `P1-05` Primary/secondary hardware acceptance tests pass on target profiles.
- [x] `P1-06` Implementation behavior matches closed decision log.
- [x] `P1-07` Quantitative gates pass on release-candidate workload report.
- [x] `P1-08` Incident/rollback/support drills completed with evidence.
- [x] `P2-09` Matrix profiles covered by repeatable test runs.
- [x] `P2-10` SBOM/signing/attribution artifacts attached to release candidate.
- [ ] `P0-HW-11` Real GPU transfer gate passed with repeatable round-trip artifacts.
- [ ] `P1-HW-12` Service-path hardware orchestration and NVML-delta evidence validated.
- [ ] `P1-HW-13` Release-facing reports/gates distinguish simulation validation from hardware execution validation.
