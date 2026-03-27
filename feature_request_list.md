# Feature Request List: TurboQuant CUDA Integration

> **Source**: llama-cpp-turboquant-cuda fork analysis
> **Target**: AstraWeave (unified_windows) codebase
> **Hardware Target**: 32 GB VRAM dGPU, 128 GB system RAM, 16 CPU cores
> **Date**: 2026-03-27
> **Goal**: Enable GPU-accelerated inference at 1.7–2.1 bits per weight via TurboQuant CUDA kernels, integrated with AstraWeave's memory orchestration layer.

---

## Critical Finding

The llama-cpp-turboquant-cuda fork registers two new GGML quantization types:

| Type | GGML ID | Bits/Weight | Block Size | Block Bytes | Values |
|------|---------|-------------|------------|-------------|--------|
| TQ1_0 | 34 | 1.6875 | 256 | 42 | Ternary {-1, 0, 1} |
| TQ2_0 | 35 | 2.0625 | 256 | 66 | 2-bit polar {-1, 0, 1} |

**These types have CPU reference implementations but NO CUDA kernels.** All GPU inference falls back to CPU for TQ-quantized tensors, negating GPU throughput entirely. This feature list closes that gap and wires the result into AstraWeave.

---

## Tier 0 — CUDA Kernel Implementation (llama-cpp-turboquant-cuda)

These are upstream contributions to the fork. Without them, nothing else matters.

### F-001: TQ2_0 CUDA Dequantization Kernel
- **Priority**: P0 (Critical)
- **File**: `ggml/src/ggml-cuda/convert.cu`
- **Description**: Implement `dequantize_block_tq2_0` CUDA kernel. TQ2_0 uses simple 2-bit packing (4 values per byte), making it the easier kernel to implement first.
- **Algorithm**: For each byte in `block_tq2_0.qs`, extract 4 × 2-bit values via shift-and-mask, compute `(q - 1) * d` where `d = __half2float(block.d)`.
- **Thread layout**: 32 threads per block, each thread handles 8 elements (256 / 32).
- **Validation**: Compare output against CPU `dequantize_row_tq2_0` reference.
- **Expected impact**: Enables GPU-resident TQ2_0 tensors.

### F-002: TQ1_0 CUDA Dequantization Kernel
- **Priority**: P0 (Critical)
- **File**: `ggml/src/ggml-cuda/convert.cu`
- **Description**: Implement `dequantize_block_tq1_0` CUDA kernel. TQ1_0 uses base-3 ternary encoding (5 trits per byte in `qs`, 4 trits per byte in `qh`).
- **Algorithm**: For each byte, compute trit extraction via `(byte * pow3[n]) / 243` then `(trit * 3) >> 8` to recover {0,1,2}, then `(val - 1) * d`.
- **Complexity**: Higher than TQ2_0 due to base-3 unpacking. Consider precomputed LUT in shared memory (243 entries × 5 outputs = 1215 bytes, fits easily).
- **Thread layout**: 32 threads, each decodes 8 elements. LUT loaded cooperatively.
- **Validation**: Bitwise match against CPU `dequantize_row_tq1_0`.

### F-003: TQ2_0 Vector Dot Product CUDA Kernel
- **Priority**: P0 (Critical)
- **File**: `ggml/src/ggml-cuda/vecdotq.cuh`
- **Description**: Implement `vec_dot_tq2_0_q8_1` for matrix-vector multiplication. This is the hot path for decode (token generation).
- **Algorithm**: Unpack 2-bit values, subtract 1 to get {-1,0,1}, multiply with Q8_1 quantized activations, accumulate. Ternary multiply simplifies to conditional add/subtract/skip.
- **Constants**: Define `VDR_TQ2_0_Q8_1_MMVQ` and `VDR_TQ2_0_Q8_1_MMQ`.
- **Register to**: `ggml/src/ggml-cuda/mmvq.cu` dispatch switch.
- **Expected throughput**: Ternary ops should be faster than Q4_K vec_dot (fewer multiplies).

### F-004: TQ1_0 Vector Dot Product CUDA Kernel
- **Priority**: P0 (Critical)
- **File**: `ggml/src/ggml-cuda/vecdotq.cuh`
- **Description**: Implement `vec_dot_tq1_0_q8_1`. Same ternary {-1,0,1} semantics as TQ2_0 but with base-3 unpacking overhead.
- **Optimization**: Precompute sign lookup for each byte (243 possible byte values → 5 signs each). Store as shared memory LUT. Ternary dot product: `sum += sign[i] * q8[i]` (conditional add/sub/nop).
- **Register to**: `ggml/src/ggml-cuda/mmvq.cu` dispatch switch.

### F-005: TQ1_0/TQ2_0 MMQ (Matrix-Multiply Quantized) Kernels
- **Priority**: P1 (High)
- **File**: `ggml/src/ggml-cuda/mmq.cu`
- **Description**: Implement tile-based matrix multiplication for TQ types. MMQ kernels are used for batch processing (prefill) and are critical for prompt throughput.
- **Approach**: Follow the pattern of existing `mmq_mxfp4` — load TQ blocks into shared memory tiles, dequantize on the fly, accumulate with FMA.
- **Tile size**: 256 elements (QK_K) aligns naturally with CUDA warp multiples.

### F-006: Flash Attention Support for TQ KV Cache Types
- **Priority**: P1 (High)
- **Files**: `ggml/src/ggml-cuda/fattn-tile.cuh`, `fattn-common.cuh`
- **Description**: Enable TQ1_0 and TQ2_0 as valid `type_k` and `type_v` for Flash Attention kernels. Currently, quantized V cache requires Flash Attention, but FA doesn't dispatch TQ types.
- **Approach**: Add TQ dequantization paths to the FA tile loader. For KV cache specifically, TQ2_0 is preferred (simpler unpacking in the attention inner loop).
- **Gate**: Requires F-001 and F-002 first.
- **Build flag**: Compile under `GGML_CUDA_FA_ALL_QUANTS=ON`.

### F-007: MMVQ Dispatch Registration for TQ Types
- **Priority**: P0 (Critical, but trivial once F-003/F-004 exist)
- **File**: `ggml/src/ggml-cuda/mmvq.cu`
- **Description**: Add `case GGML_TYPE_TQ1_0:` and `case GGML_TYPE_TQ2_0:` to the MMVQ dispatch switch, calling the kernels from F-003/F-004.
- **Also update**: `ggml/src/ggml-cuda/ggml-cuda.cu` type support checks.

---

## Tier 1 — AstraWeave Quantization Framework Updates

These changes update AstraWeave's orchestration layer to be aware of real TurboQuant capabilities.

### F-010: Update SimulatedTurboQuantProvider with Real Compression Ratios
- **Priority**: P1 (High)
- **File**: `astrawave/quantization.py`
- **Description**: The current `SimulatedTurboQuantProvider` assumes `32.0 / bit_width` compression. Real TQ types have specific block overhead:
  - TQ1_0: 42 bytes / 256 elements = 1.6875 bpw → ratio = 32.0 / 1.6875 = **18.96x** (not 9.14x at 3.5-bit)
  - TQ2_0: 66 bytes / 256 elements = 2.0625 bpw → ratio = 32.0 / 2.0625 = **15.52x**
- **Action**: Add `TQ1_0` and `TQ2_0` as distinct quantization backends alongside the generic TurboQuant simulation. Update `supported_bit_widths()` to return `(1.6875,)` and `(2.0625,)` respectively.
- **Breaking change**: The 3.5-bit default no longer applies. Callers using `SimulatedTurboQuantProvider` should migrate to the specific providers.

### F-011: Add TQ1_0 and TQ2_0 Backend Providers
- **Priority**: P1 (High)
- **File**: `astrawave/quantization.py`
- **Description**: Create `TQ1_0Provider` and `TQ2_0Provider` classes implementing `QuantizationProvider` protocol, with exact compression ratios derived from the GGML block structures.
- **Metadata**: Include `block_size: 256`, `block_bytes: 42|66`, `ggml_type_id: 34|35`, `encoding: "ternary_base3"|"2bit_polar"`.

### F-012: Update Tier-Provider Mapping for 32 GB VRAM
- **Priority**: P1 (High)
- **File**: `astrawave/quantization.py`
- **Description**: Current mapping: HOT→TurboQuant, WARM→FP8, COLD→None. With 32 GB VRAM and real TQ kernels:
  - **HOT (VRAM)**: TQ2_0 for KV cache (simpler kernel, lower decode overhead, 15.5x compression)
  - **WARM (pinned RAM)**: TQ1_0 for overflow KV blocks (maximum compression, 19x, acceptable CPU-side decode cost)
  - **COLD (pageable RAM)**: FP8 or None (depending on whether data will be promoted)
- **Make configurable**: Allow tier-provider mapping to be overridden via runtime profile.

### F-013: KV Cache Type Pass-Through in Backend Options
- **Priority**: P1 (High)
- **File**: `astrawave/runtime_tuning.py`
- **Description**: Add `type_k` and `type_v` fields to backend_options, mapping to llama.cpp's `llama_context_params.type_k` / `type_v`. This allows AstraWeave to instruct the backend to use TQ-quantized KV caches.
- **Values**: `"f16"` (default), `"f32"`, `"q8_0"`, `"q4_0"`, `"tq1_0"`, `"tq2_0"`.
- **Validation**: Reject `type_v` quantization unless `flash_attn` is also enabled (llama.cpp constraint).

### F-014: Flash Attention Enable Flag in Backend Options
- **Priority**: P1 (High)
- **File**: `astrawave/runtime_tuning.py`
- **Description**: Add `flash_attn: true|false|"auto"` to backend_options. Required for quantized V cache (F-013). Default to `"auto"` which lets llama.cpp decide based on device capability.

---

## Tier 2 — Runtime Tuning & Throughput Profile

### F-020: New "throughput" Runtime Profile
- **Priority**: P1 (High)
- **File**: `astrawave/runtime_tuning.py`
- **Description**: Add `THROUGHPUT_RUNTIME_PROFILE = "throughput"` alongside existing `default` and `vram_constrained`. This profile is optimized for maximum tok/s on systems with ample VRAM (≥24 GB).
- **Backend options for throughput profile**:
  ```python
  {
      "num_ctx": 8192,        # Large context for prefill amortization
      "num_batch": 512,       # Saturate GPU compute
      "num_gpu": -1,          # All layers on GPU
      "num_thread": 16,       # Max CPU threads for any offloaded work
      "flash_attn": "auto",   # Enable flash attention
      "type_k": "tq2_0",     # TQ2_0 KV cache keys (when CUDA kernels available)
      "type_v": "f16",       # F16 values (safest with FA)
      "offload_kqv": True,   # KV cache on GPU
  }
  ```

### F-021: Raise Large-Model Threshold for 32 GB VRAM
- **Priority**: P2 (Medium)
- **File**: `astrawave/runtime_tuning.py`
- **Description**: `is_large_model()` currently triggers at 14B. On 32 GB VRAM, models up to ~34B (Q4_K_M) fit fully GPU-resident. Make the threshold configurable or VRAM-budget-aware.
- **Approach**: `is_large_model(size, vram_budget_gb)` where threshold scales: 14B at 8GB, 34B at 32GB, 70B at 80GB.

### F-022: Model-Size-Aware Backend Option Scaling
- **Priority**: P2 (Medium)
- **File**: `astrawave/runtime_tuning.py`
- **Description**: `_vram_constrained_backend_options()` has hardcoded thresholds (14B, 30B, 70B) tuned for 8 GB VRAM. Add parallel `_throughput_backend_options()` with 32 GB-tuned values:
  - <14B: num_ctx=32768, num_batch=512
  - 14B–34B: num_ctx=8192, num_batch=256
  - 34B–70B: num_ctx=4096, num_batch=128 (with TQ2_0 KV cache)
  - >70B: num_ctx=2048, num_batch=64 (with TQ1_0 KV cache, partial CPU offload)

---

## Tier 3 — Tiering & Fallback Policy Updates

### F-030: Reduce Hot Headroom for High-VRAM Systems
- **Priority**: P2 (Medium)
- **File**: `astrawave/tiering.py`
- **Description**: `PlacementPolicy.hot_headroom_ratio` is 0.20 (20%). On 32 GB VRAM this reserves 6.4 GB unused. Reduce to 0.05–0.10 for throughput profile, keeping more tensors GPU-resident.
- **Approach**: Make headroom ratio a function of VRAM budget: `max(0.05, min(0.20, 4.0 / vram_budget_gb))`. At 8 GB → 0.20 (safe). At 32 GB → 0.125. At 80 GB → 0.05.

### F-031: Quantization-Aware Tier Placement
- **Priority**: P2 (Medium)
- **File**: `astrawave/tiering.py`
- **Description**: `PlacementPlanner.classify()` accepts `hot_compression_ratio` but doesn't know *which* quantization type achieves it. With TQ CUDA kernels, the compression ratio in HOT tier changes from simulated ~9x to real 15.5x (TQ2_0) or 19x (TQ1_0).
- **Action**: Pass the actual `QuantizationResult` metadata (including kernel availability) into placement decisions. If CUDA kernel exists → allow HOT placement at TQ compression. If no kernel → force WARM/COLD (CPU fallback would kill throughput).

### F-032: Fallback Ladder — KV Quantization Upgrade Specifics
- **Priority**: P2 (Medium)
- **File**: `astrawave/fallback.py`
- **Description**: `FallbackStep.KV_QUANTIZATION_UPGRADE` is abstract. Define the concrete progression:
  1. F16 → TQ2_0 (15.5x compression, moderate quality impact)
  2. TQ2_0 → TQ1_0 (19x compression, higher quality impact)
  3. TQ1_0 → context reduction (next ladder step)
- **Metadata**: Each step should carry the target `type_k`/`type_v` and expected memory savings.

### F-033: Throughput-Optimized Fallback Timings
- **Priority**: P3 (Low)
- **File**: `astrawave/fallback.py`
- **Description**: Current anti-oscillation: 30s cooldown, 15s dwell. For throughput workloads on high-VRAM systems, faster recovery is preferred:
  - Throughput profile: 10s cooldown, 5s dwell, churn window 15s.
  - Stability profile: keep current 30s/15s/30s defaults.
- **Gate**: Only apply reduced timings when `runtime_profile == "throughput"`.

---

## Tier 4 — Hardware Detection & Capability Probing

### F-040: TurboQuant CUDA Kernel Availability Probe
- **Priority**: P1 (High)
- **File**: `astrawave/hardware_probe.py`
- **Description**: Detect whether the active llama.cpp/Ollama binary was compiled with TQ CUDA kernel support. Without this, AstraWeave cannot safely recommend TQ KV cache types.
- **Approach**:
  1. Query Ollama's `/api/show` endpoint for model info including supported quantization types.
  2. Attempt a small TQ2_0-quantized test inference and check if it runs on GPU (not CPU fallback).
  3. Cache the result for session lifetime.
- **Fallback**: If probe fails, assume TQ CUDA kernels are NOT available and use simulated compression only.

### F-041: VRAM Budget Auto-Detection
- **Priority**: P2 (Medium)
- **File**: `astrawave/hardware_probe.py`
- **Description**: Current VRAM budget defaults to 8 GB via `ASTRAWEAVE_VRAM_BUDGET_BYTES`. Auto-detect actual VRAM from nvidia-smi / NVML and set budget to `total_vram * 0.90` (reserve 10% for OS/display).
- **Override**: Environment variable still takes precedence for manual tuning.

### F-042: Compute Capability Detection for Flash Attention
- **Priority**: P2 (Medium)
- **File**: `astrawave/hardware_probe.py`
- **Description**: Flash Attention requires CUDA compute capability ≥ 7.5 (Turing+). Probe GPU compute capability and gate `flash_attn: "auto"` recommendations accordingly.
- **Source**: NVML `nvmlDeviceGetCudaComputeCapability()` or parse from nvidia-smi.

---

## Tier 5 — Build, Distribution & Ollama Integration

### F-050: Custom llama.cpp Build with TQ CUDA Kernels
- **Priority**: P1 (High — required for all CUDA kernel features)
- **Description**: Build the llama-cpp-turboquant-cuda fork with CUDA support enabled:
  ```bash
  cmake -B build \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_FA=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DGGML_CUDA_GRAPHS=ON \
    -DCMAKE_CUDA_ARCHITECTURES="native" \
    -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j16
  ```
- **Output**: `llama-server`, `llama-quantize`, `llama-bench` binaries with TQ CUDA support.
- **Deliverable**: Build script in `scripts/build_turboquant.sh` (or `.ps1` for Windows).

### F-051: Ollama Custom Backend Integration
- **Priority**: P1 (High)
- **Description**: Ollama bundles its own llama.cpp. To use TQ CUDA kernels, either:
  1. **Option A**: Replace Ollama's bundled llama.cpp with our fork's build (risky, version-locked).
  2. **Option B**: Run `llama-server` from our fork directly, bypassing Ollama. AstraWeave's `OllamaInferenceRuntime` already speaks the same HTTP API.
  3. **Option C**: Contribute TQ CUDA kernels upstream to llama.cpp, then wait for Ollama to pick them up.
- **Recommendation**: Option B for development/benchmarking. Option C for production.
- **Action**: Add `LlamaCppServerInferenceRuntime` adapter to `astrawave/inference_runtime.py` that connects directly to a `llama-server` instance (same API as Ollama but without the model management layer).

### F-052: GGUF Model Conversion to TQ Formats
- **Priority**: P2 (Medium)
- **Description**: Convert existing GGUF models to TQ1_0 or TQ2_0 format using `llama-quantize`:
  ```bash
  llama-quantize input.gguf output-tq2.gguf TQ2_0
  llama-quantize input.gguf output-tq1.gguf TQ1_0
  ```
- **Importance matrix**: Both TQ types support `--imatrix` for activation-aware quantization, which improves quality significantly at sub-2-bit precision.
- **Deliverable**: Script `scripts/convert_to_tq.sh` that handles download → quantize → validate pipeline.

### F-053: Windows-Native Build Script
- **Priority**: P2 (Medium)
- **Description**: PowerShell build script for Windows 11 targeting MSVC + CUDA Toolkit:
  ```powershell
  cmake -B build -G "Visual Studio 17 2022" -A x64 `
    -DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON `
    -DCMAKE_CUDA_ARCHITECTURES="native"
  cmake --build build --config Release -j 16
  ```
- **File**: `scripts/build_turboquant.ps1`

---

## Tier 6 — Benchmarking & Validation

### F-060: TQ Throughput Benchmark Suite
- **Priority**: P1 (High)
- **File**: New script or extension of `scripts/ram_target_benchmark.py`
- **Description**: Benchmark matrix comparing tok/s across quantization types on the target hardware:
  - Baseline: Q4_K_M (current default)
  - Comparison: TQ2_0, TQ1_0, Q8_0, F16
  - KV cache variants: F16 KV, TQ2_0 KV, Q4_0 KV
  - Context lengths: 2048, 4096, 8192, 16384, 32768
  - Batch sizes: 32, 128, 256, 512
- **Output**: TSV with columns: `quant_type, kv_type, num_ctx, num_batch, tok_per_sec, peak_vram_mb, time_to_first_ms`

### F-061: Quality Validation (Perplexity)
- **Priority**: P2 (Medium)
- **Description**: Measure perplexity degradation from TQ quantization using `llama-perplexity` on a standard corpus (WikiText-2). Establish quality floor:
  - TQ2_0 perplexity should be within 5% of Q4_K_M on same model.
  - TQ1_0 perplexity should be within 15% of Q4_K_M.
  - If worse, flag for importance-matrix re-quantization.

### F-062: CUDA Kernel Correctness Tests
- **Priority**: P0 (Critical — gate for all CUDA kernel work)
- **Description**: Unit tests comparing CUDA kernel output against CPU reference:
  - Dequantization: `dequantize_block_tq{1,2}_0` CUDA vs CPU, max abs error < 1e-6.
  - Vec dot: `vec_dot_tq{1,2}_0_q8_1` CUDA vs CPU, max relative error < 1e-4.
  - End-to-end: Full model inference (8B) produces identical logits (top-5 match) between CPU and CUDA TQ paths.

### F-063: Memory Pressure Regression Tests
- **Priority**: P2 (Medium)
- **File**: Extension of existing test suite
- **Description**: Verify that TQ KV cache types reduce peak VRAM as expected:
  - TQ2_0 KV at 8192 context uses ≤ 50% of F16 KV memory.
  - TQ1_0 KV at 8192 context uses ≤ 35% of F16 KV memory.
  - No OOM across 100 iterations with TQ2_0 KV + throughput profile on 32 GB VRAM.

---

## Tier 7 — Documentation & Governance

### F-070: Update plan.md Decision Log
- **Priority**: P3 (Low)
- **Description**: Add decision D-006 documenting TurboQuant CUDA integration strategy and the choice of TQ2_0 as default HOT-tier KV cache quantization.

### F-071: Update docs/compatibility-matrix.md
- **Priority**: P3 (Low)
- **Description**: Add Profile P-E: "32 GB dGPU + 128 GB RAM + TurboQuant CUDA" with expected tok/s ranges, supported model sizes, and KV cache configurations.

### F-072: Update docs/api-contract.md
- **Priority**: P3 (Low)
- **Description**: Document new backend_options fields (`type_k`, `type_v`, `flash_attn`, `num_gpu`, `num_thread`) and the `throughput` runtime profile.

---

## Dependency Graph

```
F-062 (correctness tests)
  ├── F-001 (TQ2_0 dequant kernel)
  ├── F-002 (TQ1_0 dequant kernel)
  ├── F-003 (TQ2_0 vec_dot kernel)
  └── F-004 (TQ1_0 vec_dot kernel)
        ├── F-007 (MMVQ dispatch) ─────────┐
        └── F-005 (MMQ kernels)            │
              └── F-006 (FA TQ support)    │
                                           ▼
F-050 (custom build) ◄────────────── All CUDA kernels
  ├── F-051 (Ollama integration)
  └── F-052 (GGUF conversion)
        └── F-060 (benchmark suite)
              └── F-061 (quality validation)

F-010 (real compression ratios) ◄── CUDA kernel analysis
  ├── F-011 (TQ providers)
  │     └── F-012 (tier mapping)
  ├── F-013 (KV type pass-through)
  │     └── F-014 (flash_attn flag)
  └── F-020 (throughput profile)
        ├── F-021 (large-model threshold)
        ├── F-022 (backend option scaling)
        ├── F-030 (headroom reduction)
        ├── F-031 (quant-aware placement)
        ├── F-032 (fallback KV progression)
        └── F-033 (throughput fallback timings)

F-040 (TQ probe) ◄── F-050 (custom build)
F-041 (VRAM auto-detect) — independent
F-042 (compute cap detection) — independent
```

---

## Implementation Order (Recommended)

**Phase 1 — CUDA Foundation** (1-2 weeks):
1. F-001, F-002 (dequantization kernels)
2. F-003, F-004 (vec_dot kernels)
3. F-007 (MMVQ dispatch)
4. F-062 (correctness tests)
5. F-050 (build script)

**Phase 2 — Integration Bridge** (1 week):
6. F-010, F-011 (real TQ providers in AstraWeave)
7. F-013, F-014 (backend options pass-through)
8. F-051 (direct llama-server adapter)
9. F-052 (model conversion scripts)

**Phase 3 — Throughput Optimization** (1 week):
10. F-020, F-022 (throughput profile)
11. F-030, F-031 (tiering updates)
12. F-032 (fallback KV progression)
13. F-060 (benchmark suite)

**Phase 4 — Advanced CUDA** (1 week):
14. F-005 (MMQ kernels)
15. F-006 (Flash Attention TQ support)
16. F-040, F-041, F-042 (hardware probing)

**Phase 5 — Polish** (ongoing):
17. F-021, F-033 (tuning refinements)
18. F-061, F-063 (quality + regression tests)
19. F-070, F-071, F-072 (documentation)
