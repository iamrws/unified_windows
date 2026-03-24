# AstraWeave Phase 4: Live Inference Bridge CTO Spec

## Executive Summary

Phase 4 turns AstraWeave from an orchestration prototype into a practical local-inference bridge for Windows machines with large system RAM and a small dGPU.

The goal is not to replace model runtimes. The goal is to make AstraWeave the control plane that can:

- start and own a local session,
- load a model intent,
- route prompt execution to a local inference backend,
- preserve security, privacy, and typed-error behavior,
- and make 128 GB host RAM usable for real local model runs without pretending the GPU has to hold the entire model.

This phase should be treated as the bridge between the current service prototype and a real local LLM workflow.

## What This Phase Is

Phase 4 is the first production-oriented bridge for real prompt execution on the target machine.

It should prove that:

- the service lifecycle can carry a real model identity,
- the SDK/CLI can reach a local runtime backend,
- prompt execution can succeed on a local model host,
- and the user can see honest, structured results without raw prompt logging by default.

The intended operating model is:

1. AstraWeave manages session, policy, security, and observability.
2. A local inference backend such as Ollama or llama.cpp performs prompt generation.
3. AstraWeave coordinates the request and captures structured results.

## What This Phase Is Not

Phase 4 is not:

- distributed inference,
- remote model hosting,
- a cloud API proxy,
- prompt/content telemetry collection,
- automatic model downloading,
- or a claim that AstraWeave itself already contains a full tensor runtime.

## Core Architecture

### Control Plane

AstraWeave remains responsible for:

- caller identity checks,
- session lifecycle,
- policy selection,
- local-only telemetry defaults,
- error taxonomy,
- and structured status reporting.

### Runtime Adapter Layer

Add a backend abstraction that can route prompt execution to a local model host.

Minimum supported backend:

- `ollama`

Recommended future backend:

- `llama.cpp`

### Session and Prompt Flow

The operational flow should be:

`CreateSession -> LoadModel -> RunStep(prompt) -> CloseSession`

The service should carry the model name and prompt metadata, but raw prompt content must not be added to telemetry by default.

### Machine Fit Assumption

The target machine is a Windows host with:

- 128 GB system RAM,
- about 8 GB of VRAM,
- and a local model backend able to spill most of the model into host RAM when needed.

The point of the phase is to let the host RAM do real work.

## Milestones

### Milestone 1: Backend Bridge

- Add a local inference backend abstraction.
- Support at least one backend that can answer prompts on the local machine.
- Preserve current service/session behavior for callers that do not pass prompt data.

### Milestone 2: API Surface

- Extend service, IPC, SDK, and CLI to carry prompt requests and runtime selection.
- Keep older call paths working.
- Keep errors typed and stable.

### Milestone 3: Operator Runbook

- Document a copy-paste workflow for starting the local service and running a prompt.
- Make the model/backend requirements explicit.
- Keep the instructions honest about what is and is not service-native yet.

### Milestone 4: Validation

- Add a smoke script that validates service startup, model load, and prompt execution against a local backend.
- Verify no prompt content is emitted into telemetry by default.
- Verify the happy path on the target Windows class of machine.

## Acceptance Gates

Phase 4 should not be treated as done until all of the following are true:

- A local session can be created and a model can be loaded through AstraWeave.
- A prompt can be executed against a local backend and return a structured result.
- The default path remains local-only.
- Raw prompt and completion text are not written to telemetry by default.
- Typed errors still propagate through IPC/SDK/CLI.
- The smoke workflow can be run by an operator without editing source code.

## Key Risks

- Backend drift: Ollama and llama.cpp have different request/response conventions.
- Memory pressure: a large model can fit in 128 GB RAM but still stall if the backend uses a poor quantization or context setting.
- User confusion: if the bridge is too clever, people may think AstraWeave itself already owns full inference execution.
- Privacy regression: prompt text can leak into logs if the bridge reuses generic debug output.
- Hardware variability: the 8 GB dGPU is helpful, but model choice still has to respect VRAM limits.

## Out Of Scope

- Distributed serving.
- Multi-GPU scheduling.
- Automatic model search/download.
- Cloud fallback.
- Batch inference orchestration.
- Benchmark leaderboard work.
- Cross-process prompt history syncing.

## Operator Success Definition

For an operator, success in this phase looks like:

- they run one command,
- the service comes up,
- a model is loaded,
- a prompt is answered locally,
- and the output tells the truth about which part was handled by AstraWeave versus the local backend.

That is the right bar for this phase.

## Phase 5 Continuity

Phase 4 proves the bridge. Phase 5 adds the large-model control layer for 14B+ workloads by making runtime profiles and backend tuning more explicit.

The operator guide for that next step is in `docs/phase5-large-model-orchestration.md`.
