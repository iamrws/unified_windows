# AstraWeave Throughput Optimization Program

## Preamble

This document adapts the autonomous experimentation loop architecture from
`autoresearch` to AstraWeave's memory orchestration domain. Where the original
optimizes validation loss (val_bpb) on a fixed time budget by mutating a
training script, this program optimizes **tokens per second** (tok/s) on a
fixed hardware budget by mutating AstraWeave's orchestration policy, tier
placement, quantization strategy, and backend configuration.

The target system:

| Resource       | Specification                        |
|----------------|--------------------------------------|
| GPU VRAM       | 32 GB dGPU (e.g. RTX 4090 / A100)   |
| System RAM     | 128 GB DDR5                          |
| CPU            | 16 cores / 32 threads                |
| OS             | Windows 11 Pro                       |
| Backend        | llama-server (TurboQuant CUDA fork) or Ollama |

The information-theoretic framing: we are searching the configuration space
C = (tier_policy, quantization_params, kv_cache_type, backend_options, concurrency_params)
for the point c* that maximizes the throughput functional T(c) = tokens / wall_seconds,
subject to the constraint that peak VRAM never exceeds 32 GB and no OOM occurs.

### TurboQuant CUDA Kernel Integration

This program incorporates the `llama-cpp-turboquant-cuda` fork, which adds two
ultra-low-bit quantization formats to the GGML type system:

| Format | GGML ID | Bits/Weight | Block Size | Block Bytes | Encoding |
|--------|---------|-------------|------------|-------------|----------|
| TQ1_0  | 34      | 1.6875      | 256        | 42          | Ternary base-3 (5 trits/byte) |
| TQ2_0  | 35      | 2.0625      | 256        | 66          | 2-bit polar (4 values/byte) |

**Critical status**: CPU reference implementations exist; **CUDA kernels do not yet
exist** and must be implemented (see `feature_request_list.md` F-001 through F-007).
Until CUDA kernels land, TQ-quantized tensors fall back to CPU, negating GPU throughput.

**Compression impact on 32 GB VRAM** (compared to F16 baseline):

| Quantization | 30B Model Size | KV Cache (8192 ctx, 32 layers) | Freed VRAM |
|-------------|----------------|--------------------------------|------------|
| F16         | ~60 GB         | ~10.7 GB                       | 0 GB       |
| Q4_K_M      | ~17 GB         | ~10.7 GB (F16 KV)              | 0 GB       |
| Q4_K_M + TQ2_0 KV | ~17 GB  | ~0.69 GB                       | **~10 GB** |
| Q4_K_M + TQ1_0 KV | ~17 GB  | ~0.56 GB                       | **~10.1 GB** |
| TQ2_0 weights | ~3.8 GB      | ~0.69 GB (TQ2_0 KV)            | **~27 GB** |

The freed VRAM enables: larger context windows, bigger batch sizes, or fitting
models that would otherwise require CPU offload — all of which directly increase tok/s.

See `feature_request_list.md` for the complete implementation roadmap.

---

## 1. Setup

To set up a new optimization run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar27`).
   The branch `throughput/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b throughput/<tag>` from current main.
3. **Read the in-scope files** for full context:
   - `plan.md` -- normative spec, closed decisions, ownership boundaries.
   - `feature_request_list.md` -- TurboQuant CUDA integration roadmap. **Reference.**
   - `astrawave/runtime_tuning.py` -- profile selection, backend option computation. **Primary mutation target.**
   - `astrawave/tiering.py` -- HOT/WARM/COLD classification, headroom policy. **Primary mutation target.**
   - `astrawave/quantization.py` -- compression providers, tier-aware selection. **Primary mutation target.**
   - `astrawave/fallback.py` -- ladder order, anti-oscillation thresholds. **Secondary mutation target.**
   - `astrawave/service.py` -- session lifecycle, pressure thresholds. **Read-only reference** (mutations here risk correctness).
   - `astrawave/inference_runtime.py` -- Ollama/llama-server adapter. **Secondary mutation target.**
   - `scripts/ram_target_benchmark.py` -- benchmark harness. **Read-only.**
   - `llama-cpp-turboquant-cuda/` -- TurboQuant CUDA fork. **Reference for CUDA kernel work.**
4. **Verify inference backend is running**:
   - **Option A (Ollama)**: `curl -s http://127.0.0.1:11434/api/tags | python -m json.tool`
   - **Option B (llama-server with TQ CUDA)**: Build the fork first (see `feature_request_list.md` F-050),
     then start: `./llama-server -m <model.gguf> -ngl -1 --host 127.0.0.1 --port 8080`
   Option B is required for TQ-quantized models and KV cache experiments.
5. **Select the target model**: Default to the largest model that fits in 32 GB VRAM.
   For TQ experiments, prepare both standard and TQ-quantized GGUF variants:
   ```bash
   # Standard baseline
   qwen2.5:32b-instruct-q4_K_M
   # TQ variants (convert with llama-quantize)
   llama-quantize model-f16.gguf model-tq2.gguf TQ2_0
   llama-quantize model-f16.gguf model-tq1.gguf TQ1_0
   ```
   Record the exact model tag -- all experiments within a phase must use the
   same model for comparability.
6. **Initialize results.tsv**: Create `results.tsv` with the header row only.
7. **Confirm and go**.

---

## 2. The Metric

The **sole optimization target** is:

```
tok_per_sec = total_output_tokens / wall_seconds
```

measured end-to-end through AstraWeave's `RunStep` path (session creation through
response completion). This captures the full pipeline cost: tier placement decisions,
quantization overhead, backend option resolution, Ollama HTTP round-trip, and
llama.cpp inference.

Secondary metrics to record but NOT optimize directly:

| Metric            | Purpose                                      |
|-------------------|----------------------------------------------|
| `peak_vram_mb`    | Hard constraint: must stay <= 32768 MB       |
| `peak_ram_mb`     | Soft constraint: should stay <= 96000 MB     |
| `time_to_first_ms`| Time to first token (latency, not throughput)|
| `p95_step_ms`     | RunStep p95 latency                          |
| `cpu_util_pct`    | Average CPU utilization during inference     |

---

## 3. What You CAN Modify

These files are your experiment surface. All parameters are fair game:

### Tier A -- Primary (highest expected throughput impact)

**`astrawave/runtime_tuning.py`**:
- `_vram_constrained_backend_options()` -- context window, batch size, KV quantization flags.
  The current defaults are tuned for 8 GB VRAM. On 32 GB, substantially larger
  context windows (8192-32768) and batch sizes (128-512) are feasible and will
  directly increase tok/s by amortizing attention overhead.
- Profile thresholds -- the 14B boundary for `is_large_model()` should be
  re-evaluated for 32 GB VRAM. Models up to ~34B may run fully GPU-resident.
- Add a new `throughput` profile alongside `default` and `vram_constrained`.

**`astrawave/tiering.py`**:
- `PlacementPolicy` thresholds -- `hot_reuse_score`, `warm_reuse_score`, `hot_headroom_ratio`.
  On 32 GB VRAM, the 20% headroom reserve is conservative. Reducing to 5-10%
  keeps more tensors HOT (VRAM-resident), reducing transfer stalls.
- Placement logic ordering -- active tensors first, then by reuse score descending.

### Tier B -- Secondary (moderate expected impact)

**`astrawave/quantization.py`**:
- **Replace simulated TurboQuant with real TQ providers**: The current
  `SimulatedTurboQuantProvider` assumes 32/3.5 = 9.14x compression. Real TQ types
  achieve 18.96x (TQ1_0) and 15.52x (TQ2_0) — roughly 2x more compression than modeled.
- Add `TQ1_0Provider` and `TQ2_0Provider` with exact block-structure ratios.
- **KV cache quantization**: This is the highest-leverage TQ application. KV cache
  dominates VRAM at long contexts. TQ2_0 KV at 8192 context: ~0.69 GB vs ~10.7 GB
  at F16 — freeing ~10 GB for larger batch sizes or more model layers on GPU.
- Tier-provider mapping update for 32 GB VRAM:
  - HOT → TQ2_0 (GPU-resident, simpler 2-bit kernel, lower decode overhead)
  - WARM → TQ1_0 (maximum compression for overflow, acceptable CPU decode cost)
  - COLD → None or FP8

**`astrawave/fallback.py`**:
- Anti-oscillation thresholds: 30s cooldown is conservative for throughput
  optimization. On a system with headroom, faster recovery (10-15s) from
  fallback states can reclaim throughput sooner.
- Ladder ordering: for throughput, batch reduction before context reduction
  may preserve parallelism better (context reduction kills prefill efficiency).
- **KV quantization upgrade progression**: Define concrete steps for
  `FallbackStep.KV_QUANTIZATION_UPGRADE`: F16 → TQ2_0 → TQ1_0 → context reduction.

### Tier C -- Tertiary (smaller expected impact, worth exploring)

**`astrawave/inference_runtime.py`**:
- Ollama timeout tuning, connection reuse, request payload options.
- `num_gpu` (GPU layer count), `num_thread` (CPU thread count for offloaded layers),
  `numa` (NUMA-aware allocation) passed through backend_options.
- **New backend options for TQ CUDA integration**:
  - `type_k`: KV cache key quantization type (`"f16"`, `"q8_0"`, `"q4_0"`, `"tq2_0"`, `"tq1_0"`)
  - `type_v`: KV cache value quantization type (same options; quantized V requires flash_attn)
  - `flash_attn`: `true`/`false`/`"auto"` — required for quantized V cache
  - `offload_kqv`: `true` — keep KV cache on GPU (critical for throughput)
- **Direct llama-server adapter**: Add `LlamaCppServerInferenceRuntime` that connects
  to the TQ CUDA fork's `llama-server` directly (same OpenAI-compatible API as Ollama,
  but supports TQ-quantized models natively).

### New files (create only if needed):
- `astrawave/throughput_profile.py` -- if the throughput profile grows complex
  enough to warrant its own module. Prefer keeping it in `runtime_tuning.py`.

---

## 4. What You CANNOT Modify

- `astrawave/service.py` -- the session lifecycle and security contract are
  correctness-critical. Read it for context, do not mutate it.
- `astrawave/security.py`, `astrawave/telemetry.py` -- security and privacy invariants.
- `astrawave/ipc_server.py`, `astrawave/ipc_client.py`, `astrawave/ipc_protocol.py` -- wire contract.
- `tests/` -- do not weaken existing tests. You may ADD tests.
- `prepare.py` equivalent: there is no fixed eval script here, but the benchmark
  harness (`scripts/ram_target_benchmark.py`) is read-only.
- Do not install new packages. Zero runtime dependencies is a hard project constraint.

---

## 5. The Benchmark Harness

Each experiment is measured by running a standardized inference workload through
AstraWeave's service path. The benchmark script:

```bash
# Standard benchmark invocation
ASTRAWEAVE_VRAM_BUDGET_BYTES=$((32 * 1024 * 1024 * 1024)) \
python -m astrawave.cli serve --endpoint tcp://127.0.0.1:8765 &
SERVICE_PID=$!
sleep 3

python scripts/ram_target_benchmark.py \
    --model <TARGET_MODEL> \
    --target-ram-gb 96 \
    --iterations 10 \
    --prompt "Explain the RSA cryptosystem, including key generation, encryption, decryption, and why factoring large semiprimes is computationally hard. Then derive the correctness proof from Euler's theorem." \
    > run.log 2>&1

kill $SERVICE_PID
```

If `ram_target_benchmark.py` does not emit tok/s directly, compute it from
the output fields:

```bash
grep "^tok_per_sec:\|^peak_vram_mb:\|^run_step_p95_ms:" run.log
```

The prompt is deliberately long (multi-paragraph technical output expected) to
amortize prefill cost and measure sustained decode throughput -- the regime
where memory bandwidth is the bottleneck and our tier/quantization choices
have maximum impact.

**Time budget per experiment**: 10 minutes maximum (including service startup
and all iterations). If a run exceeds 15 minutes, kill it and treat it as a
failure.

---

## 6. Output Format

After each experiment, extract and record:

```
tok_per_sec:      <float>
peak_vram_mb:     <float>
peak_ram_mb:      <float>
time_to_first_ms: <float>
p95_step_ms:      <float>
total_tokens:     <int>
wall_seconds:     <float>
```

---

## 7. Logging Results

When an experiment is done, log it to `results.tsv` (tab-separated).

Header and columns:

```
commit	tok_per_sec	peak_vram_gb	status	description
```

1. git commit hash (short, 7 chars)
2. tok_per_sec achieved (e.g. 42.37) -- use 0.00 for crashes
3. peak VRAM in GB, rounded to .1f (e.g. 28.4) -- use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	tok_per_sec	peak_vram_gb	status	description
a1b2c3d	23.50	24.2	keep	baseline (vram_constrained profile on 32GB)
b2c3d4e	31.80	29.1	keep	increase num_ctx to 8192 and num_batch to 256
c3d4e5f	28.90	30.5	discard	TurboQuant 2.5-bit (decompression overhead dominates)
d4e5f6g	0.00	0.0	crash	num_ctx 65536 (OOM)
e5f6g7h	38.20	31.2	keep	FP8 for HOT tier + num_batch 512 + 10% headroom
```

Do NOT commit `results.tsv` -- leave it untracked.

---

## 8. The Experiment Loop

The experiment runs on a dedicated branch (e.g. `throughput/mar27`).

LOOP FOREVER:

1. **Inspect state**: current branch, last experiment result, cumulative best tok/s.
2. **Form hypothesis**: choose ONE variable to change per experiment. Examples:
   - Increase `num_ctx` from 2048 to 8192 (amortize attention setup).
   - Increase `num_batch` from 32 to 256 (saturate GPU compute units).
   - Reduce `hot_headroom_ratio` from 0.20 to 0.05 (keep more tensors in VRAM).
   - Switch HOT tier quantization from TurboQuant to FP8 (lower decode overhead).
   - Add `num_gpu: -1` to backend_options (force full GPU offload).
   - Add `num_thread: 16` to backend_options (maximize CPU parallelism for offloaded layers).
   - Increase `is_large_model` threshold from 14B to 34B (avoid unnecessary constraints).
   - Tune anti-oscillation cooldown from 30s to 10s (faster recovery).
3. **Implement**: edit the relevant file(s).
4. **Commit**: `git add <files> && git commit -m "<hypothesis>"`.
5. **Run**: execute the benchmark, redirect output: `> run.log 2>&1`.
6. **Measure**: extract tok/s and peak VRAM from run.log.
7. **If empty output**: the run crashed. `tail -n 50 run.log` for the stack trace.
   Fix if trivial; otherwise log as crash and revert.
8. **Record**: append to results.tsv.
9. **Decision**:
   - If tok/s improved AND peak_vram_mb <= 32768: **keep**. Advance the branch.
   - If tok/s equal or worse, OR VRAM exceeded: **discard**. `git reset --hard HEAD~1`.
10. **Repeat**.

### Experiment Prioritization (Throughput Hierarchy)

Based on the information-theoretic analysis of where throughput is lost in the
AstraWeave pipeline, and incorporating TurboQuant CUDA kernel availability,
prioritize experiments in this order:

**Phase 1 -- Backend Saturation** (highest expected ROI):
1. Maximize `num_batch` until VRAM-limited (GPU compute utilization).
2. Maximize `num_ctx` until VRAM-limited (prefill amortization).
3. Set `num_gpu: -1` to force all layers onto GPU (eliminate PCIe transfers).
4. Set `num_thread: 16` for any CPU-offloaded work.
5. Enable `flash_attn: "auto"` (fused attention kernel, 2-3x faster).
6. Set `offload_kqv: true` (KV cache on GPU, not CPU).

**Phase 2 -- TurboQuant KV Cache** (highest expected VRAM savings):
> **Gate**: Requires TQ CUDA kernels (F-001 through F-007 in feature_request_list.md).
> If kernels are not yet available, skip to Phase 3 and return here later.
7. Set `type_k: "tq2_0"` — quantize KV cache keys to TQ2_0 (15.5x compression).
   This alone frees ~9 GB of VRAM at 8192 context, enabling larger batches.
8. Set `type_v: "tq2_0"` — quantize KV cache values (requires flash_attn=true).
   Combined with step 7, KV cache drops from ~10.7 GB to ~0.69 GB.
9. With freed VRAM, increase `num_ctx` from 8192 to 16384 or 32768.
10. With freed VRAM, increase `num_batch` from 512 to 1024 or 2048.
11. Test `type_k: "tq1_0"` + `type_v: "tq2_0"` (asymmetric: max K compression,
    moderate V compression for quality preservation in attention scores).

**Phase 3 -- Memory Policy Tuning**:
12. Reduce `hot_headroom_ratio` (more VRAM for model weights).
13. Raise `is_large_model` threshold to 34B (avoid premature constraint).
14. Update AstraWeave quantization providers with real TQ compression ratios.
15. Test all-HOT tier strategy (no WARM/COLD) for models that fit in 32 GB.

**Phase 4 -- Pipeline Optimization**:
16. Reduce anti-oscillation cooldown (faster fallback recovery).
17. Reorder fallback ladder (batch reduction after precision reduction).
18. Define KV quantization fallback progression: F16 → TQ2_0 → TQ1_0.

**Phase 5 -- TurboQuant Weight Quantization** (model-level compression):
> **Gate**: Requires MMQ CUDA kernels (F-005) for competitive prefill throughput.
19. Test TQ2_0-quantized model weights (2.06 bpw, ~3.8 GB for 30B model).
    This fits a 70B model in 32 GB VRAM (vs impossible at Q4_K_M).
20. Test TQ1_0-quantized model weights (1.69 bpw) for extreme compression.
21. Measure perplexity impact — if quality is acceptable, this unlocks
    previously impossible model sizes on 32 GB hardware.

**Phase 6 -- Compound Effects**:
22. Combine the best settings from Phases 1-5.
23. Fine-tune the combined configuration (binary search on batch size, context length).
24. Ablation: remove individual TQ components to measure marginal contribution.
25. Test asymmetric configurations: TQ2_0 weights + TQ1_0 KV, or vice versa.

### Simplicity Criterion

All else being equal, simpler is better. A 0.5 tok/s improvement that adds
30 lines of routing logic? Probably not worth it. A 0.5 tok/s improvement
from deleting an unnecessary headroom reserve? Definitely keep. A 2 tok/s
improvement from a clean new throughput profile? Keep.

The goal is to find the configuration c* such that T(c*) is maximized with
minimal policy complexity. Occam's razor applied to systems engineering.

---

## 9. Hardware-Specific Optimization Notes

### Memory Bandwidth Analysis

On the target system (32 GB VRAM, PCIe 4.0 x16 = ~32 GB/s host-device bandwidth):

- **Decode phase** is memory-bandwidth-bound. Each token requires reading all
  model weights once. For a 30B Q4_K_M model (~17 GB), the theoretical maximum
  decode throughput is:
  ```
  tok/s_max = memory_bandwidth / model_size = 1000 GB/s / 17 GB ~ 59 tok/s
  ```
  (Assuming HBM2e on A100; ~1 TB/s. On RTX 4090 with GDDR6X: ~1 TB/s.)

- **Prefill phase** is compute-bound. Larger batch sizes amortize the compute
  cost per token.

- **KV cache** grows with context length. At FP16, each layer costs:
  ```
  kv_per_layer = 2 * num_heads * head_dim * seq_len * 2 bytes
  ```
  For a 32-layer model with 40 heads, head_dim=128, seq_len=8192:
  ```
  kv_total = 32 * 2 * 40 * 128 * 8192 * 2 = ~10.7 GB
  ```

  **TurboQuant KV cache compression** (the single highest-leverage optimization):

  | KV Type | Bytes/Element | KV Total (8192 ctx) | Savings vs F16 | Freed VRAM |
  |---------|---------------|---------------------|----------------|------------|
  | F16     | 2.0           | 10.7 GB             | —              | —          |
  | Q8_0    | 1.0625        | 5.7 GB              | 47%            | 5.0 GB     |
  | Q4_0    | 0.5625        | 3.0 GB              | 72%            | 7.7 GB     |
  | TQ2_0   | 0.2578        | 1.38 GB             | 87%            | 9.3 GB     |
  | TQ1_0   | 0.1641        | 0.88 GB             | 92%            | 9.8 GB     |

  At 32768 context (4x), TQ2_0 KV uses ~5.5 GB vs ~42.8 GB at F16 — the
  difference between fitting in VRAM and catastrophic OOM. This is where
  TurboQuant CUDA kernels have transformative throughput impact: by compressing
  KV cache, they free VRAM budget for larger context windows (better prefill
  amortization) and larger batch sizes (better GPU utilization), both of which
  directly increase tok/s.

  **Constraint**: Quantized V cache requires Flash Attention enabled.
  Quantized K cache works without FA. Recommended: `type_k: "tq2_0"` always;
  `type_v: "tq2_0"` only with `flash_attn: true`.

### CPU Thread Allocation

With 16 cores / 32 threads:
- Reserve 2 threads for AstraWeave service + IPC.
- Reserve 2 threads for OS / Ollama daemon overhead.
- Allocate remaining 12-28 threads to inference backend via `num_thread`.
- Experiment with `num_thread` values: 12, 16, 24, 28 to find the throughput
  plateau (diminishing returns from hyperthreading contention).

### NUMA Considerations

On a single-socket 16-core system, NUMA is not a factor. But if the system
is dual-socket, pass `numa: true` in backend_options to enable NUMA-aware
memory allocation (reduces cross-socket memory access latency).

---

## 10. Convergence and Termination

**NEVER STOP**. Once the experiment loop begins, do NOT pause to ask the human
if you should continue. The human may be away and expects you to work
autonomously until manually interrupted.

If you run out of simple single-variable experiments:
- **Combine**: merge the best-performing individual changes.
- **Binary search**: find the exact threshold where a parameter starts hurting
  (e.g. the largest num_batch before VRAM overflows).
- **Ablate**: remove components of the best configuration one at a time to find
  the minimal effective configuration.
- **Reread**: go back to `service.py` and `inference_runtime.py` for new angles.
- **TQ kernel analysis**: Read the CUDA kernel implementations in
  `llama-cpp-turboquant-cuda/ggml/src/ggml-cuda/` for optimization ideas.
  The ternary {-1,0,1} values in TQ formats enable specialized dot product
  kernels that replace multiply-accumulate with conditional add/subtract —
  potentially faster than Q4_K vec_dot on the same hardware.
- **Asymmetric quantization**: Try different types for K vs V caches
  (K is more tolerant of aggressive quantization than V in attention scoring).
- **Model size frontier**: With TQ2_0 weights (~2 bpw), a 70B model fits in
  ~11.5 GB — leaving 20 GB for KV cache and compute buffers. Test whether
  a larger, more capable TQ-quantized model outperforms a smaller Q4_K model
  in both quality and throughput.
- **Radical changes**: consider whether a completely different tier strategy
  (e.g. all-HOT with no WARM/COLD for models that fit in 32 GB) eliminates
  tier-transition overhead entirely.
- **ISWA opportunity**: For models supporting Interleaved Sliding Window Attention,
  the SWA cache can be sized to `window_size * n_streams + ubatch_size` —
  dramatically smaller than full context. Combined with TQ KV, this could
  enable 128K+ context on 32 GB VRAM.

The loop runs until the human interrupts you. At 10 minutes per experiment,
you can run ~6/hour, ~48 over an 8-hour session.
