# AGENTS.md

## Purpose
This file captures durable session context so future agents can continue work without losing intent, standards, or decisions.

## Product Direction Locked In
- v1 focus is reliability-first on Windows local inference.
- Primary target path is dGPU-first (8 GB VRAM class).
- `CacheCoherentUMA` is a secondary optimized path and must not block v1.
- Telemetry is local-only by default; any export is explicit opt-in.
- Service + SDK model is in scope; CLI-only product mode is not the main v1 path.

## Source-of-Truth Documents
- `plan.md` is the normative baseline spec.
- `problems.md` is the release-gate register.
- Supporting governance and contract docs live under `docs/`.

## Major Architecture Choices Implemented
- Core runtime modules are present in `astrawave/` (service, fallback, security, telemetry, IPC server/client/protocol, SDK, CLI, service host).
- IPC contract is now centralized in `astrawave/ipc_protocol.py` as canonical wire schema and validation logic.
- `astrawave/ipc_server.py` keeps compatibility wrappers (`Ipc*` envelope names), but dispatch and validation now rely on canonical protocol behavior.
- Real CUDA transfer primitives now live in `astrawave/cuda_runtime.py` and are reused by `scripts/cuda_transfer_poc.py`.
- Service `RunStep` now supports feature-flagged hardware mode (`ASTRAWEAVE_RUNSTEP_MODE=hardware` or `ASTRAWEAVE_ENABLE_HARDWARE_RUNSTEP=1`) and returns `run_mode` plus `hardware_result`.

## Security and Trust-Boundary Decisions
- IPC requires explicit caller identity by default at server boundary.
- Caller identity is authorized before service dispatch.
- Per-connection caller binding is enforced: caller switching on one connection is denied.
- Optional transport authkey alignment is supported via `ASTRAWEAVE_IPC_AUTHKEY` (server and client).
- Same-user default authorization and explicit cross-user allowlist semantics remain enforced by `SecurityGuard`.

## CLI and Runtime Behavior Decisions
- CLI backend policy is explicit: `--backend {auto,remote,local}`.
- Default behavior now prefers real remote IPC path (`auto`) instead of silently defaulting to local simulator behavior.
- Local simulator backend remains available but is explicit opt-in (`--backend local` or `local://...` endpoint).
- Connection failures now return typed JSON API errors (no traceback leakage).

## Testing and Validation Pattern
- Execution pattern used: targeted contract and e2e tests first, then full-suite run.
- Current validated state: `python -m unittest discover -s tests -v` passes with 62/62 tests (`reports/runlogs/unittest_rc_2026-03-24.txt`).
- New and expanded tests cover:
- explicit caller enforcement,
- per-connection caller-binding rejection,
- canonical protocol request handling in server,
- CLI remote-default behavior and explicit local backend selection,
- localhost-only bind enforcement,
- IPC request payload size cap (1 MiB),
- security-deny telemetry emission,
- typed JSON error behavior on malformed endpoints,
- capability-matrix profile mapping (`P-A` through `P-D`),
- service single-flight `RunStep` conflict rejection.
- hardware-mode `RunStep` executor invocation and degraded-on-runtime-failure behavior,
- release-gate split reporting (`simulation_ready` vs `hardware_ready`).
- Soak evidence (March 23-24, 2026):
- `reports/runlogs/summary_2026-03-23_235145.json` (`200/200` pass, `run_step_p95_ms=194.063`),
- `reports/runlogs/summary_2026-03-23_235729.json` (`1000/1000` pass, `run_step_p95_ms=191.373`),
- `reports/runlogs/summary_2026-03-24_001253.json` (strict single-caller run shows expected `AW_ERR_RATE_LIMITED` abuse-control behavior),
- `reports/runlogs/summary_rc_w7b_chat_2026-03-24.json` (`W-7B-CHAT`, `100/100` pass, `run_step_p95_ms=181.811`),
- `reports/runlogs/summary_rc_w13b_chat_2026-03-24.json` (`W-13B-CHAT`, `100/100` pass, `run_step_p95_ms=183.845`).
- Automation harnesses:
- `scripts/soak_test.ps1` (unit-test + soak + GPU logging + JSON summary artifacts),
- `scripts/run_operations_drills.py` (ops drills and evidence report),
- `scripts/run_hardware_gate.py` (service-triggered hardware-mode RunStep + NVML delta gate),
- `scripts/generate_compliance_artifacts.py` (SBOM/attribution/checksum/signing artifacts),
- `scripts/generate_release_gate_report.py` (threat + matrix + workload + final readiness synthesis).
- Hardware gate artifact: `reports/release_gate/hardware_gate_2026-03-24.json` (`pass`).
- Consolidated gate verdict artifact: `reports/release_gate/release_gate_readiness_2026-03-24.json` (`hardware_ready` with split track status).

## Documentation Alignment Decisions
- `README.md`, `docs/api-contract.md`, `docs/security-architecture.md`, `docs/threat-model.md`, and `reports/phase3_audit_report.md` were updated to match implemented behavior.
- `README.md` now explicitly documents hardware probe boundary, CUDA transfer PoC, hardware gate script, and split readiness tracks.
- `problems.md` now contains reopened hardware gates (`P0-HW-11`, `P1-HW-12`, `P1-HW-13`) and linked March 24, 2026 hardware evidence artifacts.

## Known Residual Risk
- Full OS-native caller attestation at transport boundary is still a future hardening step; current control set reduces spoofing risk but does not fully replace native peer-identity verification.
- Hardware mode currently executes a transfer proof path, but full model-aware tensor orchestration policies are still maturing.

## Collaboration Preferences from This Session
- User preference is aggressive execution over partial analysis: "fix everything then continue until finished."
- User prefers multi-agent orchestration framing for large phases (manager/orchestrator model, parallelized work, continued progress until closure).
- User expects periodic status updates and forward momentum without unnecessary pauses.
- User is comfortable with ambitious exploration, but final output must be concrete, test-backed, and implemented.
- User requires GitHub to be updated on every completed implementation cycle for this repository.

## Practical Continuation Rules for Future Agents
- Treat `plan.md` and `problems.md` as contractual gates.
- Preserve canonical IPC schema in `ipc_protocol.py`; avoid reintroducing duplicate protocol logic.
- Do not make local simulator behavior the silent default path for production-like CLI usage.
- Keep security behavior explicit and typed-error-based at boundaries.
- Before declaring completion, run full tests and ensure docs and report stay synchronized with code.
- Before declaring completion, commit all relevant changes and push them to GitHub (`origin/main`) unless the user explicitly says not to push.
