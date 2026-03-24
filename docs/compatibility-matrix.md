# AstraWeave Compatibility and Support Matrix (v1)

## 1. Capability Modes

Capability modes used by runtime policy:

- `NUMA_dGPU`: discrete GPU plus system RAM (primary v1 path).
- `CacheCoherentUMA`: cache-coherent integrated path (secondary optimized path).
- `UMA`: integrated shared memory without full coherency (supported, stricter safeguards).
- `UNSUPPORTED`: fail fast with actionable diagnostics.

## 2. Minimum Platform Requirements

- OS: Windows 11 23H2 or newer.
- Driver model: WDDM 3.1 or newer.
- API baseline: DirectX 12 compatible GPU + driver stack.
- RAM minimum for v1 acceptance: 32 GB.
- VRAM minimum for primary release gate: 8 GB discrete GPU.

## 3. Support Matrix

| Profile | Example Hardware Class | Capability Mode | Support Level | Expected Behavior |
| --- | --- | --- | --- | --- |
| P-A (Primary) | 8 GB+ NVIDIA/AMD dGPU + >=32 GB RAM | `NUMA_dGPU` | Supported and release-gated | Reliability-first ladder, bounded stalls, no hard OOM under gate workload |
| P-B (Secondary) | Intel/AMD integrated with cache-coherent UMA | `CacheCoherentUMA` | Supported, secondary gate | Lower-copy fast path, same API guarantees |
| P-C (Degraded) | UMA without full coherency | `UMA` | Supported with degraded-mode contract | More conservative prefetch and stricter churn controls |
| P-D (Not Supported) | Legacy DX12-incomplete or WDDM below minimum | `UNSUPPORTED` | Fail fast | Clear unsupported diagnostics and no partial startup |

## 4. Degraded-Mode Contract

When degraded behavior is active:

- Runtime may reduce aggressive prefetching.
- Runtime may switch to stability profile automatically.
- Runtime may lower concurrency and batch ceilings.
- Runtime must emit explicit reason code and mode flag.

## 5. Acceptance Mapping

Acceptance test requirements:

- Primary release gate workload must pass on `P-A`.
- Secondary capability branch tests must pass on `P-B`.
- Degraded-mode behaviors and reason codes must pass on `P-C`.
- Unsupported checks must fail fast on `P-D` with deterministic diagnostics.

