# AstraWeave Phase 6: RAM-Target Benchmarking and Compression-Aware Tuning

## Why This Phase Exists

If the goal is to run models that can push host RAM toward 90 GB on a machine with limited VRAM, we need repeatable evidence, not one-off runs.

Phase 6 adds a benchmark/autotune sweep that:

- runs multiple option candidates,
- measures memory and throughput,
- scores candidates against a RAM target,
- and writes reports that can be shared.

## Research Framing

This phase treats optimization as a measurable system problem:

1. **Fit objective**: get as close as possible to target host RAM use without instability.
2. **Performance objective**: keep decode throughput acceptable.
3. **Reproducibility objective**: produce consistent reports for later comparison.

## Compression Framing (Practical)

Quantization controls model weight size. Runtime options control live memory behavior.

- Lower-bit quantization (for example Q4) usually improves fit margin.
- Higher-bit quantization (for example Q6/Q8) can improve quality but often increases memory pressure.
- Context size (`num_ctx`) and batching (`num_batch`) can heavily increase runtime memory usage.

This benchmark uses **compression-aware option profiles** as candidates (Q4/Q6/Q8 style hints), then scores them by observed behavior.

## What Was Implemented

- New script: `scripts/ram_target_benchmark.py`
- New tests: `tests/test_ram_target_benchmark.py`
- Report outputs:
  - JSON: `reports/benchmarks/ram_target_<run-id>.json`
  - Markdown: `reports/benchmarks/ram_target_<run-id>.md`

## Primary Metrics

Per candidate:

- success rate across iterations,
- decode tokens/sec (`eval_tokens_per_second`),
- end-to-end tokens/sec,
- peak observed Ollama private memory (GB),
- peak observed Ollama working set (GB),
- RAM delta vs target (GB).

## Directions: How To Run

## 1) Ensure Ollama Is Running

```powershell
Invoke-RestMethod http://127.0.0.1:11434/api/tags
```

If this returns model info, Ollama is reachable.

## 2) Pull the Model You Want to Benchmark

Example:

```powershell
ollama pull qwen2.5:14b
```

For very high RAM targets, test a larger model tag when available.

## 3) Run RAM-Target Benchmark Sweep

```powershell
cd C:\Users\grzeg\Documents\Jito\testmarch24\unified_windows
python scripts/ram_target_benchmark.py --model qwen2.5:14b --target-ram-gb 90 --runtime-profile vram_constrained --iterations 2 --cooldown-seconds 2 --max-tokens 128 --ipc-timeout-seconds 900
```

The script prints JSON with:

- best candidate,
- JSON report path,
- Markdown report path.

## 4) Read the Report

Open the generated markdown report under:

- `reports/benchmarks/`

Use:

- `peak_private_gb` and `ram_target_delta_gb` for fit,
- `avg_eval_tokens_per_second` for speed.

## 5) Move Toward 90 GB Safely

If observed RAM is too low:

- increase model size class,
- or increase `num_ctx` candidate pressure by testing larger models first.

If runs are unstable:

- lower `num_ctx`,
- lower `num_batch`,
- keep `vram_constrained`.

## Recommended Validation Standard

Before claiming optimization progress:

1. Run at least 3 benchmark sweeps on the same model/settings.
2. Compare median metrics, not best single run.
3. Keep historical reports in `reports/benchmarks/` for regression tracking.
