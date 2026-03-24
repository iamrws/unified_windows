# AstraWeave (v1 Spec and Runtime Workspace)

AstraWeave is a Windows user-mode service plus SDK that orchestrates RAM and VRAM as one logical pool for local LLM inference, with reliability-first behavior on 8 GB dGPU systems.

This repository contains the normative v1 spec, supporting governance docs, and the current Python runtime/service prototype.

Phase 2 introduced the local IPC transport layer plus SDK and CLI wrappers around the core service prototype.

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
- `docs/phase4-live-inference-ctospec.md`

## Phase 2 Runtime

- `astrawave/service.py`: in-memory service prototype with security, telemetry, and fallback integration.
- `astrawave/ipc_protocol.py`: JSON request/response envelope contract.
- `astrawave/ipc_server.py`: local IPC adapter over the service prototype.
- `astrawave/ipc_client.py`: transport client for pipe/TCP endpoints.
- `astrawave/sdk.py`: convenience wrapper around the IPC client.
- `astrawave/cli.py`: JSON-first command-line entrypoint.

## Hardware Probe Boundary

The new `hardware-probe` CLI command is the first place where AstraWeave reads real NVIDIA platform data:

- `nvidia-smi` is parsed when the tool is present.
- NVML is queried through `ctypes` when NVIDIA's library is available.
- The command still returns a JSON envelope even when those tools are missing, so the probe is safe to run on non-NVIDIA systems.

There is also a real transfer proof script:

- `scripts/cuda_transfer_poc.py` uses the CUDA Driver API (`nvcuda.dll`) through `ctypes` to perform a real host->device->host byte copy and verify integrity.
- `scripts/run_hardware_gate.py` triggers hardware-mode `RunStep` in `AstraWeaveService` and checks for service-triggered NVML memory delta evidence.

The service orchestration layer is still intentionally bounded:

- service/session/tiering behavior remains in-process and deterministic.
- residency changes, fallback steps, and migration latency are controlled by the Python runtime.
- the hardware gate currently validates real transfer behavior, while full model-aware tensor orchestration remains a separate future milestone.

## Phase 4 Live Inference Bridge

Phase 4 is the bridge for testing a real local model on a Windows host with large system RAM and a small dGPU.

What it does today:

- starts AstraWeave's local service for session orchestration,
- loads a model name into the service,
- sends a real prompt to a local inference backend such as Ollama,
- and prints a structured JSON report.

What it does not do yet:

- it does not make AstraWeave itself the prompt-generation engine,
- it does not hide the local backend choice,
- and it does not log raw prompt or completion text by default.

Example workflow:

```powershell
# Terminal 1: make sure Ollama is running locally
ollama serve

# Pull any local model you want to test. A 7B or 8B instruct model is a sensible start.
ollama pull <your-local-model-tag>

# Terminal 2: run the AstraWeave bridge smoke
python scripts/live_inference_smoke.py --runtime-backend ollama --model <your-local-model-tag> --prompt "Explain in one paragraph why 128 GB of RAM helps local LLM inference."
```

The smoke script starts the AstraWeave service itself, loads the chosen model name with the chosen runtime backend, and sends the prompt through AstraWeave `RunStep`. It is the fastest honest test of the current bridge phase.

Note: the smoke path imports the AstraWeave package, so it assumes the runtime import path is healthy.

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
# Real NVIDIA diagnostics:
python -m astrawave.cli hardware-probe
# Real CUDA transfer PoC (host->device->host round-trip):
python scripts/cuda_transfer_poc.py --bytes 1048576 --pretty
# Service-path hardware gate (RunStep hardware mode + NVML delta check):
python scripts/run_hardware_gate.py --run-id 2026-03-24
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
# Optional explicit hardware gate path:
python scripts/generate_release_gate_report.py --run-id 2026-03-24 --hardware-gate-report reports/release_gate/hardware_gate_2026-03-24.json
```

The consolidated readiness report now publishes split status:

- `simulation_ready`: contract/governance/reliability track status.
- `hardware_ready`: adds hardware execution gates (`P0-HW-11`, `P1-HW-12`, `P1-HW-13`).
