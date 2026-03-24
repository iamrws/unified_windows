# thoughts_1

## Euclid: Apple/Windows Overlap Analysis

From an Apple and Windows systems perspective, the overlap is strongest when UMA is treated as a stack of three layers: shared physical memory, cache/coherency behavior, and a developer-facing runtime model.

### What Transfers Cleanly

- Transferable ideas come from the architecture, Metal programming model, and Windows reverse-engineering sections.
- Windows equivalents are capability-specific: DirectX 12 on iGPU with `UMA` and `CacheCoherentUMA`, Vulkan host-visible + device-local memory types, and managed memory approaches in CUDA/ROCm.
- The practical pattern is consistent: reduce copies, keep data resident where possible, use persistent mappings when the platform supports them, and apply explicit synchronization where needed.

### Where Apple and Windows Diverge

- Apple's model is not only "shared RAM"; the differentiator is the integrated stack: on-package memory, shared cache behavior, hardware coherency, and tightly integrated software.
- Windows UMA on APUs can approximate parts of the programming model but usually not the same latency/bandwidth envelope or pointer-sharing behavior.
- Discrete GPU paths on Windows (including ReBAR/managed memory approaches) still face migration and interconnect costs.

### Practical Implications

- Build a capability-based abstraction instead of an "Apple vs Windows" abstraction.
- Branch behavior by coherency and residency semantics:
- `CacheCoherentUMA`: closest to coherent shared-pool behavior.
- `UMA` without full coherency: lower-copy is possible, but synchronization and CPU-read behavior are less ideal.
- Discrete GPU migration model: treat transfers as first-class costs.
- Important distinction: zero-copy, shared address space, and hardware coherency are related but not equivalent.

## Descartes: Product Outline + Mind Map

### Product Outline

**Working name:** Unified Memory AI Runtime for Windows

**Purpose**

- Let local AI apps treat system RAM and GPU VRAM as one managed pool.
- Reduce model load friction, OOM failures, and manual device-placement work.
- Enable larger local models on consumer Windows hardware without heavy hand-tuning.

**Primary users**

- Windows AI app developers.
- Power users running local model tools.
- OEM/platform teams improving local AI consistency.

**User value**

- Run models larger than VRAM-only capacity by spilling inactive data to RAM.
- Improve launch reliability by adapting to available memory.
- Reduce custom paging/offload code.
- Graceful degradation under pressure rather than abrupt failure.

**Goals**

- Single logical memory abstraction over RAM + VRAM.
- Keep critical data in fast memory while minimizing stalls.
- Preserve system responsiveness.
- Make behavior observable and debuggable.
- Align with Windows platform primitives.

**Non-goals**

- Not a total replacement for framework memory management.
- Not guaranteed transparent zero-copy everywhere.
- Not a "magic speed boost" feature.
- Not a bypass of vendor drivers or Windows memory policy.

**Architecture**

- Host orchestrator tracks allocations, hotness, and pressure.
- Unified memory manager applies priority/residency policy.
- GPU access layer targets DX12/WDDM-compatible resource behavior.
- Transfer scheduler moves data between RAM and VRAM based on demand.
- Telemetry/diagnostics expose residency, stalls, faults, and fallbacks.

**Memory model**

- Hot: active attention buffers, current weights, KV cache, immediate tensors.
- Warm: soon-needed weights, reusable embeddings, context windows.
- Cold: infrequently used layers, idle sessions, archived state.

**Scheduling and tiering**

- Predictive prefetch for upcoming execution phases.
- Cold-first eviction from VRAM.
- Reserve headroom for transient spikes.
- Auto-shift from performance mode to stability mode under pressure.
- Support framework priority hints (latency vs throughput).

**DX12/WDDM considerations**

- Prefer explicit DX12-compatible memory/resource management.
- Respect WDDM residency budgets and scheduling.
- Do not treat pageable VRAM like CPU RAM.
- Handle iGPU and dGPU paths separately.
- Assume eviction can occur and support rehydration.

**Constraints**

- PCIe/transfer bandwidth cost.
- Page fault latency spikes.
- Fragmentation/alignment challenges.
- Driver and device variability.
- RAM pressure can destabilize overall system performance.

**Safety/fallback**

- Reduce batch size, cache window, or precision before failing.
- Fall back selected layers to RAM residency or CPU execution when needed.
- Enter stability mode to avoid oscillation under heavy churn.
- Disable advanced path on unsupported hardware/driver behavior.
- Always prioritize OS responsiveness.

**Roadmap**

- Phase 1: abstraction + conservative policy.
- Phase 2: DX12/WDDM-aware residency + diagnostics.
- Phase 3: predictive tiering + adaptive fallbacks.
- Phase 4: integrate major local inference runtimes.
- Phase 5: multi-GPU and NUMA-aware placement.
- Phase 6: OEM/end-user automation and polish.

**Success metrics**

- Larger models launch on more Windows devices.
- Fewer OOM crashes/manual tuning steps.
- Better cold-start latency than naive CPU fallback.
- Stable UI responsiveness during inference.
- Increased developer adoption and lower support burden.

### Mind Map

```text
Windows Unified Memory AI System
├─ Goal
│  ├─ RAM + VRAM as one managed pool
│  ├─ Run larger local models
│  └─ Reduce OOMs and manual tuning
├─ Users
│  ├─ AI app developers
│  ├─ Power users
│  └─ OEM/platform teams
├─ Core Architecture
│  ├─ Host orchestrator
│  ├─ Unified memory manager
│  ├─ GPU access layer
│  ├─ Transfer scheduler
│  └─ Telemetry/diagnostics
├─ Memory Tiers
│  ├─ Hot
│  ├─ Warm
│  └─ Cold
├─ Scheduling
│  ├─ Predictive prefetch
│  ├─ Cold-first eviction
│  ├─ Headroom reservation
│  ├─ Adaptive batch sizing
│  └─ Stability mode
├─ Platform
│  ├─ DX12
│  ├─ WDDM residency rules
│  ├─ Driver variability
│  └─ Multi-adapter handling
├─ Safety
│  ├─ Lower precision if needed
│  ├─ Reduce batch size
│  ├─ RAM/CPU fallback
│  └─ Preserve OS responsiveness
└─ Roadmap
   ├─ Abstraction + policy
   ├─ Residency + diagnostics
   ├─ Predictive tiering
   ├─ Framework integration
   ├─ Multi-GPU / NUMA
   └─ OEM-ready polish
```

## Parfit: Product Spec Summary

### Executive Summary

Unified Memory AI Runtime for Windows is a reliability-first platform layer that presents RAM and VRAM as one logical pool for local AI inference. Its primary value is running larger models with predictable behavior under pressure, not pretending transfer costs disappear.

### Problem Statement

Windows local inference faces fragmented memory domains, WDDM residency limits, hardware/driver variability, and framework-specific paging complexity that leads to brittle behavior and OOM/stall failure modes.

### Goals

- Single logical memory abstraction across RAM + VRAM.
- Keep active model state close to compute to minimize stalls.
- Preserve OS responsiveness.
- Make policy and fallback decisions observable.
- Integrate with DX12/WDDM platform behavior.

### Non-goals

- Not replacing every framework allocator.
- Not guaranteeing zero-copy for all hardware.
- Not bypassing OS/driver memory policy.

### Core Requirements

- Tiered classification and policy-driven placement.
- Developer hints for priority/residency.
- Demand-based migration between RAM/VRAM.
- Safe degradation under pressure.
- Clear diagnostics for movement, eviction, and fallback.

### Architecture

- Host orchestrator.
- Unified memory manager.
- GPU access layer.
- Transfer scheduler.
- Telemetry and diagnostics pipeline.

### Memory Tiering Policy

- Hot: keep in VRAM when possible.
- Warm: pre-position based on predicted use.
- Cold: candidate for eviction first.
- Reserve headroom to absorb transient decode/batch spikes.
- Switch to stability mode when transfer churn is high.

### DX12 / WDDM / Platform Assumptions

- Explicit DX12 resource handling.
- WDDM budgeting and eviction respected.
- Distinct iGPU vs dGPU strategy.
- Rehydration path required after eviction.
- Align with driver/OS control instead of fighting it.

### Fallback Behavior

- Reduce batch size/cache windows/precision before hard failure.
- Shift selected layers to RAM or CPU execution when budgets are exceeded.
- Enter anti-oscillation stability mode when migration churn grows.
- Disable advanced tiering on unsupported hardware/driver paths.

### Observability

- Residency by tier/device.
- Transfer volume and latency.
- Eviction and rehydration events.
- Pressure signals and fallback triggers.
- Human-readable explanations for policy decisions.

### Milestones

- M1: abstraction + conservative policy.
- M2: residency controls + diagnostics.
- M3: predictive prefetch/tiering + adaptive fallback.
- M4: framework/model-format integrations.
- M5: multi-GPU + NUMA-aware placement.
- M6: OEM/end-user polish.

### Success Metrics

- More successful large-model launches.
- Lower OOM rate and manual tuning burden.
- Better cold start than naive CPU-only fallback.
- Stable UI responsiveness during inference.
- Higher integration adoption.

### Risks

- Migration cost can erase gains for bandwidth-bound workloads.
- Page faults can spike latency.
- RAM overcommit can hurt system stability.
- Fragmentation may limit large tensor placement.
- Driver variability may produce inconsistent behavior.
- Poor policy can oscillate and degrade UX.

### Open Questions

- Runtime-vs-framework policy boundary.
- Amount of developer tunability vs automation.
- Minimum hardware bar for advanced tiering.
- Default telemetry surface for developers and OEMs.
