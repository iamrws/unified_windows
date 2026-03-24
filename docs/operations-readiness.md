# AstraWeave Operations Readiness (v1)

## 1. Incident Response Model

Incident severities:

- `S1`: data exposure risk, hard service outage, or repeat hard OOM in release path.
- `S2`: major reliability regression with available workaround.
- `S3`: minor reliability or diagnostics issue without user-blocking impact.

Response targets:

- `S1`: acknowledge within 15 minutes, mitigation plan within 1 hour.
- `S2`: acknowledge within 1 hour, mitigation plan within 1 business day.
- `S3`: triage in normal sprint flow.

## 2. Rollback Policy

Mandatory rollback triggers:

- Any `S1` issue confirmed in canary or release.
- Hard OOM count > 0 in 100-run release gate workload.
- `RunStep` migration-induced p95 stall > 250 ms across 3 consecutive 30-minute windows.
- Fallback oscillation > 1 ladder step per 30 seconds after stability mode entry.

Rollback mechanics:

- Keep previous signed release artifact available for immediate rollback.
- Rollback decision owner: Release Commander.
- Rollback action and reason must be logged in release journal.

## 3. Crash Artifact Standard

Required crash artifacts:

- Service minidump.
- Last 10 minutes of structured event stream (redacted).
- Process and driver version manifest.
- Session policy profile and mode (`NUMA_dGPU`, `CacheCoherentUMA`, `UMA`).
- Correlation ids for last completed and failed calls.

Forbidden in crash artifacts:

- Raw prompt/output text.
- Secrets/credential material.

## 4. Ownership Matrix

Required on-call ownership:

- Runtime core owner: fallback, scheduler, tier engine.
- Platform owner: WDDM/DX12 integration, capability detection.
- SDK owner: Python parity and client error mapping.
- Security owner: auth, threat-model closure, privacy policy enforcement.
- Release owner: signing, rollback, compliance checklist.

## 5. Failure Drill Cadence

Required drills before v1 external release:

- One auth abuse drill.
- One WDDM budget contraction and recovery drill.
- One rollback drill from candidate to previous stable artifact.

Evidence requirements:

- Drill report with timeline, action owner, observed vs expected behavior, and remediations.

## 6. Release Gate Requirements

v1 operational gate is closed only when:

- All mandatory drills are completed with published reports.
- Rollback path validated on current release candidate.
- Crash artifact pipeline verified to include all required fields.

