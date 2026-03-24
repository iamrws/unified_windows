# AstraWeave API Contract (v1)

## 1. IPC Transport and Wire Contract

Phase 2 exposes the core runtime over a local IPC boundary.

Transport behavior:

- Windows named pipes are the preferred transport on Windows.
- Local TCP socket fallback is used when named pipes are unavailable or when tests/configuration request it.
- The client and server both use `multiprocessing.connection` semantics for transport compatibility.
- The IPC layer remains local-only by default and does not expose a network service in v1.

Request envelope:

- `id`: non-empty string request id.
- `method`: non-empty string service method name.
- `params`: JSON-serializable object.
- `caller`: caller object with `user_sid` and `pid` (required by default at the server boundary).

Success response:

- `id`: request id.
- `ok`: `true`.
- `result`: JSON-serializable result or `null`.

Error response:

- `id`: request id.
- `ok`: `false`.
- `error.code`: stable `AW_*` error code.
- `error.message`: human-readable message.

Example request:

```json
{"id":"req-00000001","method":"CreateSession","params":{},"caller":{"user_sid":"S-1-5-21-1000","pid":1001}}
```

Example success response:

```json
{"id":"req-00000001","ok":true,"result":"<session-id>"}
```

Example error response:

```json
{"id":"req-00000001","ok":false,"error":{"code":"AW_ERR_AUTH_DENIED","message":"caller is not authorized for this local service"}}
```

Boundary behavior:

- Caller identity is always validated at the boundary before dispatch.
- The server binds a connection to the first accepted caller identity and rejects caller switching on that connection.
- Security decisions are enforced by both the IPC boundary and the service layer.
- Telemetry is local-only by default and remains redacted across the IPC boundary.
- IPC responses must use stable error codes and must not expose raw prompt/output content.

## 1. Versioning and Compatibility

Compatibility policy:

- Service and SDK use semantic versioning.
- Patch releases: bug fixes only, no contract changes.
- Minor releases: backward-compatible additive changes only.
- Major releases: allowed breaking changes with migration notes.

Compatibility rule:

- Python SDK mirrors service API behavior and error codes 1:1 for shared endpoints.

## 2. Core Types

Required types:

- `MemoryTier`: `HOT`, `WARM`, `COLD`.
- `ResidencyState`: `VRAM`, `PINNED_RAM`, `PAGEABLE_RAM`, `CPU_ONLY`.
- `PolicyProfile`: `STABILITY` (default), `THROUGHPUT` (opt-in).
- `PressureSnapshot`: budget, usage, pressure level, timestamp.
- `TransferEvent`: migration direction, bytes, latency, reason code.
- `FallbackEvent`: ladder step, trigger, result, reason code.
- `ApiError`: stable error code + machine-readable category + message.
- `SessionState`: `NEW`, `SESSION_CREATED`, `MODEL_LOADED`, `READY`, `RUNNING`, `DEGRADED`, `CLOSED`, `FAILED`.

## 3. Lifecycle Semantics

Required sequence:

`CreateSession -> LoadModel -> RegisterTensor/SetTierHint/PrefetchPlan -> RunStep -> CloseSession`

State transitions:

- `CreateSession`: `NEW -> SESSION_CREATED`.
- `LoadModel`: `SESSION_CREATED -> MODEL_LOADED -> READY`.
- `RunStep`: `READY/DEGRADED -> RUNNING -> READY/DEGRADED`.
- Fatal service error during `RunStep`: `RUNNING -> FAILED`.
- `CloseSession` from any non-closed state: `* -> CLOSED` (idempotent).

Out-of-order invocation:

- Any call that violates state preconditions must return `AW_ERR_INVALID_STATE`.

## 4. Concurrency and Cancellation

Concurrency rules:

- Per session: max one active `RunStep`.
- Per session: additional `RunStep` calls while one is active are rejected immediately with `AW_ERR_CONFLICT_RUN_IN_PROGRESS`.
- Multiple sessions: runtime behavior is deterministic and bounded by admission controls; strict fair-queue scheduling is not part of v1.

Cancellation rules:

- v1 does not expose cancellation-token parameters on service/SDK/CLI APIs.
- Client-side timeout or disconnect does not imply server-side cancellation semantics in v1.
- `AW_ERR_CANCELLED` remains reserved for a future explicit-cancellation API revision.

## 5. API Method Guarantees

| API | Preconditions | Postconditions | Timeout Guidance | Primary Errors |
| --- | --- | --- | --- | --- |
| `CreateSession` | Authorized caller, service healthy | Session created in `SESSION_CREATED` | 2s | `AW_ERR_AUTH_DENIED`, `AW_ERR_RATE_LIMITED`, `AW_ERR_RESOURCE_EXHAUSTED` |
| `LoadModel` | Valid session in `SESSION_CREATED` | Session `READY` with loaded model | 120s (model-dependent) | `AW_ERR_INVALID_STATE`, `AW_ERR_INVALID_ARGUMENT`, `AW_ERR_TIMEOUT` |
| `RegisterTensor` | Session `MODEL_LOADED` or `READY` | Tensor metadata registered | 5s | `AW_ERR_INVALID_STATE`, `AW_ERR_INVALID_ARGUMENT` |
| `SetTierHint` | Session exists, known tensor id | Hint recorded and auditable | 2s | `AW_ERR_INVALID_ARGUMENT`, `AW_ERR_NOT_FOUND` |
| `PrefetchPlan` | Session `READY` or `DEGRADED` | Prefetch scheduled or no-op with reason | 30s | `AW_ERR_TIMEOUT`, `AW_ERR_RESOURCE_EXHAUSTED` |
| `RunStep` | Session `READY` or `DEGRADED`, no active `RunStep` | Inference step result + telemetry correlation id | 60s target, configurable | `AW_ERR_CONFLICT_RUN_IN_PROGRESS`, `AW_ERR_DRIVER_BUDGET_CHANGED`, `AW_ERR_TIMEOUT` |
| `GetResidency` | Authorized caller owns session | Current session residency snapshot | 2s | `AW_ERR_NOT_FOUND`, `AW_ERR_AUTH_DENIED` |
| `GetPressure` | Authorized caller owns session | Current pressure snapshot | 2s | `AW_ERR_NOT_FOUND` |
| `SetPolicy` | Session exists, valid profile | Policy switched with reason code | 2s | `AW_ERR_INVALID_ARGUMENT`, `AW_ERR_INVALID_STATE` |
| `CloseSession` | Session exists or already closed | Session in `CLOSED` | 5s | `AW_ERR_NOT_FOUND` (optional), `AW_ERR_INTERNAL` |

## 6. Error Taxonomy

Stable error codes:

- `AW_OK`
- `AW_ERR_AUTH_DENIED`
- `AW_ERR_RATE_LIMITED`
- `AW_ERR_INVALID_ARGUMENT`
- `AW_ERR_INVALID_STATE`
- `AW_ERR_NOT_FOUND`
- `AW_ERR_TIMEOUT`
- `AW_ERR_CANCELLED`
- `AW_ERR_RESOURCE_EXHAUSTED`
- `AW_ERR_CONFLICT_RUN_IN_PROGRESS`
- `AW_ERR_DRIVER_BUDGET_CHANGED`
- `AW_ERR_UNSUPPORTED_CAPABILITY`
- `AW_ERR_INTERNAL`

Contract requirements:

- Every non-success response includes one stable error code.
- Error code meanings cannot change across minor versions.
- New error codes may be added only in minor versions with release notes.

## 7. Phase 2 Usage Patterns

Service code:

```python
from astrawave.ipc_server import AstraWeaveIpcServer

server = AstraWeaveIpcServer()
server.start()
```

Client code:

```python
from astrawave.ipc_client import AstraWeaveIpcClient
from astrawave.security import CallerIdentity

caller = CallerIdentity(user_sid="S-1-5-21-1000", pid=1001)
with AstraWeaveIpcClient(endpoint="tcp://127.0.0.1:8765", default_caller=caller) as client:
    session_id = client.CreateSession()
    client.LoadModel(session_id, "demo-model")
```

SDK code:

```python
from astrawave.sdk import AstraWeaveSDK

with AstraWeaveSDK(endpoint="auto", default_user_sid="S-1-5-21-1000", default_pid=1001) as sdk:
    session_id = sdk.CreateSession()
    sdk.RunStep(session_id, "decode")
```

CLI example:

```powershell
python -m astrawave.cli serve --endpoint tcp://127.0.0.1:8765
python -m astrawave.cli --endpoint tcp://127.0.0.1:8765 create-session
python -m astrawave.cli --endpoint tcp://127.0.0.1:8765 load-model <session-id> demo-model
python -m astrawave.cli --endpoint tcp://127.0.0.1:8765 get-pressure <session-id>
# Explicit simulator mode for local CLI-only state snapshots:
python -m astrawave.cli --backend local create-session
```
