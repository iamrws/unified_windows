# AstraWeave (Docs-First v1 Spec Workspace)

AstraWeave is a Windows user-mode service plus SDK that orchestrates RAM and VRAM as one logical pool for local LLM inference, with reliability-first behavior on 8 GB dGPU systems.

This repository currently contains normative design and governance artifacts for v1.

Phase 2 adds a local IPC transport layer plus SDK and CLI wrappers around the core service prototype.

## Core Documents

- `plan.md`: normative v1 baseline and closed decisions.
- `problems.md`: hard-gate register and readiness tracking.
- `docs/security-architecture.md`
- `docs/threat-model.md`
- `docs/api-contract.md`
- `docs/telemetry-governance.md`
- `docs/operations-readiness.md`
- `docs/compatibility-matrix.md`
- `docs/release-governance.md`
- `docs/test-plan.md`

## Phase 2 Runtime

- `astrawave/service.py`: in-memory service prototype with security, telemetry, and fallback integration.
- `astrawave/ipc_protocol.py`: JSON request/response envelope contract.
- `astrawave/ipc_server.py`: local IPC adapter over the service prototype.
- `astrawave/ipc_client.py`: transport client for pipe/TCP endpoints.
- `astrawave/sdk.py`: convenience wrapper around the IPC client.
- `astrawave/cli.py`: JSON-first command-line entrypoint.

## Historical Inputs

- `plan_v2.md`: earlier execution-plan draft.
- `thoughts_1.md`: concept exploration and product framing.
- `apple-unified-memory-architecture.md`: architecture research background.

## Quick Start

Start a local IPC server:

```python
from astrawave.ipc_server import AstraWeaveIpcServer

server = AstraWeaveIpcServer()
server.start()
```

Use the IPC client directly:

```python
from astrawave.ipc_client import AstraWeaveIpcClient
from astrawave.security import CallerIdentity

caller = CallerIdentity(user_sid="S-1-5-21-1000", pid=1001)
with AstraWeaveIpcClient(endpoint="tcp://127.0.0.1:8765", default_caller=caller) as client:
    session_id = client.CreateSession()
    client.LoadModel(session_id, "demo-model")
```

Use the SDK wrapper:

```python
from astrawave.sdk import AstraWeaveSDK
from astrawave.security import CallerIdentity

caller = CallerIdentity(user_sid="S-1-5-21-1000", pid=1001)
with AstraWeaveSDK(endpoint="auto", default_caller_identity=caller) as sdk:
    session_id = sdk.CreateSession()
    sdk.LoadModel(session_id, "demo-model")
```

Use the CLI:

```powershell
python -m astrawave.cli serve --endpoint tcp://127.0.0.1:8765
python -m astrawave.cli --endpoint tcp://127.0.0.1:8765 create-session
python -m astrawave.cli --endpoint tcp://127.0.0.1:8765 load-model <session-id> demo-model
python -m astrawave.cli --endpoint tcp://127.0.0.1:8765 run-step <session-id> --step-name decode
# Explicit local simulator mode (not the default path):
python -m astrawave.cli --backend local create-session
```

## Release-Gate Automation

Generate RC evidence artifacts:

```powershell
# 1) RC workloads (100 iterations each)
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\soak_test.ps1 -SkipUnitTests -Iterations 100 -WorkloadId W-7B-CHAT -ModelName W-7B-CHAT -TensorBytes 1048576 -RunId rc_w7b_chat_2026-03-24
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\soak_test.ps1 -SkipUnitTests -Iterations 100 -WorkloadId W-13B-CHAT -ModelName W-13B-CHAT -TensorBytes 2097152 -RunId rc_w13b_chat_2026-03-24

# 2) Full test log artifact
cmd /c "python -m unittest discover -s tests -v > reports\runlogs\unittest_rc_2026-03-24.txt 2>&1"

# 3) Operations drills
python scripts/run_operations_drills.py --run-id 2026-03-24

# 4) Compliance artifacts
python scripts/generate_compliance_artifacts.py --run-id 2026-03-24

# 5) Consolidated gate readiness report
python scripts/generate_release_gate_report.py --run-id 2026-03-24
```
