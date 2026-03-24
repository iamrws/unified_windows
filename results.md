# Results Log

## Date
- March 24, 2026

## Environment Snapshot
- OS: Windows (PowerShell)
- Repo path: `C:\Users\grzeg\Documents\Jito\testmarch24\unified_windows`
- Local backend: Ollama (`http://127.0.0.1:11434`)
- Available models observed:
  - `qwen2.5:7b`
  - `qwen2.5:14b` (pulled successfully)

## Run 1: 7B Baseline (Ollama + Auto Service Mode)

### Command
```powershell
python scripts/live_inference_smoke.py --runtime-backend ollama --model qwen2.5:7b --prompt "Explain how 128GB RAM helps local inference with only 8GB VRAM." --service-runstep-mode auto --ipc-timeout-seconds 300 --max-tokens 256
```

### Result
- `ok: true`
- `provider/backend: ollama`
- `model: qwen2.5:7b`
- `run_mode: hardware` (service hardware probe path succeeded)
- `finish_reason: length` (response truncated by token cap)
- `total_duration: 6965940100 ns` (~6.97s)

## Run 2: 14B Objective Test (Phase 5 Tuning Path)

### Command
```powershell
python scripts/live_inference_smoke.py --runtime-backend ollama --model qwen2.5:14b --runtime-profile vram_constrained --runtime-option num_ctx=4096 --runtime-option num_batch=24 --runtime-option num_gpu=24 --prompt "Explain in simple terms why 128GB RAM helps a 14B model on 8GB VRAM." --max-tokens 128 --ipc-timeout-seconds 600
```

### Result
- `ok: true`
- `provider/backend: ollama`
- `model: qwen2.5:14b`
- `runtime_profile: vram_constrained`
- effective runtime options include:
  - `num_ctx: 4096`
  - `num_batch: 24`
  - `num_gpu: 24`
  - `num_keep: 64`
  - `top_p: 0.9`
  - `repeat_penalty: 1.1`
- `run_mode: simulation` (service runstep mode defaulted to simulation for this command)
- `finish_reason: length` (response truncated by token cap)
- `total_duration: 34963067400 ns` (~34.96s)

## Conclusion
- 14B local inference succeeded on this machine using Ollama with constrained-VRAM tuning.
- This validates the intended high-RAM + low-VRAM operating path for the current phase.
