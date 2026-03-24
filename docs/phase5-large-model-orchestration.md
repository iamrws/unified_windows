# AstraWeave Phase 5: Large-Model Orchestration for 8 GB VRAM Hosts

## Executive Summary

Phase 5 is the control layer for running 14B+ local models on a Windows machine with about 8 GB of GPU memory and lots of system RAM.

The important idea is simple:

- AstraWeave does not make the GPU bigger.
- AstraWeave helps you choose settings that let the model spill work into host RAM without crashing.
- The local model runtime still does the actual generation.

This phase is about practical fit, not magic.

## What Changed In This Phase

Phase 5 adds size-aware runtime controls so the operator can tell AstraWeave how hard to push the local backend.

In plain terms, it adds:

- a runtime profile for constrained VRAM,
- backend option forwarding for the local model host,
- per-session defaults and per-step overrides,
- and clearer fallback behavior when a model is too large or a context window is too aggressive.

That means AstraWeave can now coordinate a large model more deliberately instead of treating every model like a small one.

## What This Phase Does

Phase 5 is meant to:

- accept a model tag such as `14b` or larger,
- choose a constrained-memory profile when the GPU is small,
- forward backend tuning options like context size and batch size,
- keep the session lifecycle inside AstraWeave,
- and keep typed errors visible when the backend cannot satisfy the request.

The usual good fit is:

- a 14B model that is quantized to a smaller file size,
- a backend such as Ollama or llama.cpp,
- 128 GB of system RAM,
- and a conservative context size to start.

## What This Phase Does Not Do

Phase 5 does not:

- turn 8 GB of VRAM into 24 GB,
- guarantee that every 14B+ model will fit at every context size,
- remove the need for quantization,
- add distributed serving,
- add cloud fallback,
- or hide the fact that larger models are slower on small GPUs.

If the model is too big or the context is too high, the right answer is usually to lower the runtime settings, not to keep retrying the same request.

## Tradeoffs To Expect

Running 14B+ models on this hardware is possible, but there are real tradeoffs:

- first-token latency will usually be slower,
- the first run often pays a model-load cost,
- higher context sizes use more memory,
- and aggressive GPU offload can fail even when the model itself is valid.

That is normal. The goal is not peak speed. The goal is to get a usable local model path that stays honest about memory pressure.

## Good Starting Settings

Start here for a 14B model on an 8 GB VRAM system:

- profile: `vram_constrained`
- context: `4096`
- batch: `24`
- max tokens: `128`

If that is stable, increase context slowly. If it is not stable, lower context first before changing anything else.

## Operator Workflow

The commands below assume the Phase 5 runtime controls are present in your build.

If your checkout does not recognize `--runtime-profile` or `--runtime-option`, you are on a pre-Phase-5 commit and need the implementation branch that includes these controls.

### 1. Start The Local Model Host

```powershell
ollama serve
```

If that says the port is already in use, the daemon is probably already running.

### 2. Pull A 14B Model

```powershell
ollama pull qwen2.5:14b
```

If you want a different model tag, use that instead. A 4-bit quantized 14B instruct model is a sensible place to start.

### 3. Start AstraWeave

```powershell
python -m astrawave.cli serve --endpoint tcp://127.0.0.1:8765
```

Leave that terminal open.

### 4. Create A Session

```powershell
python -m astrawave.cli --backend remote --endpoint tcp://127.0.0.1:8765 create-session
```

Copy the `session_id` from the JSON output.

### 5. Load The Model With A Constrained Profile

```powershell
python -m astrawave.cli --backend remote --endpoint tcp://127.0.0.1:8765 load-model <session-id> qwen2.5:14b --runtime-backend ollama --runtime-profile vram_constrained --runtime-option num_ctx=4096 --runtime-option num_batch=24 --runtime-option num_gpu=24
```

Notes:

- `num_ctx` is the context window. Smaller is easier on memory.
- `num_batch` affects how much work is processed at once.
- `num_gpu` is the Ollama-style offload knob. For other backends, use the equivalent offload setting.

### 6. Run One Prompt

```powershell
python -m astrawave.cli --backend remote --endpoint tcp://127.0.0.1:8765 run-step <session-id> --step-name decode --prompt "Explain in simple terms why 128 GB of RAM helps a 14B model on an 8 GB GPU." --max-tokens 128 --temperature 0.2 --runtime-option-override num_ctx=4096 --runtime-option-override num_batch=24 --runtime-option-override num_gpu=24
```

If this returns `finish_reason: length`, the answer was cut off by the token limit. Increase `--max-tokens` if you need a longer response.

### 7. Use The Smoke Script

```powershell
python scripts/live_inference_smoke.py --runtime-backend ollama --model qwen2.5:14b --runtime-profile vram_constrained --runtime-option num_ctx=4096 --runtime-option num_batch=24 --runtime-option num_gpu=24 --prompt "Explain in simple terms why 128 GB of RAM helps a 14B model on an 8 GB GPU." --max-tokens 128 --ipc-timeout-seconds 600
```

That is the fastest end-to-end check for this phase.

## How To Read The Output

The JSON output gives you the useful signals:

- `ok: true` means the call succeeded.
- `service.run_step.run_mode` tells you whether AstraWeave used the hardware path or fell back.
- `inference.result.backend` tells you which local backend actually answered.
- `inference.result.finish_reason` tells you whether the answer stopped normally or hit the token limit.
- `inference.result.raw.load_duration` is often high on the first run. That is the model loading into memory.

If the first run is slower than the second, that is expected.

## Troubleshooting

### Timeout

If you get `AW_ERR_TIMEOUT: IPC request timed out`:

- raise `--ipc-timeout-seconds` to `300` or `600`,
- keep the first prompt short,
- and wait for the backend to finish loading the model before retrying.

The first run is often slower because the backend has to load weights, build caches, and settle into memory.

### Context Too High

If the backend complains about context or memory:

- lower `--runtime-option num_ctx=4096` to `2048`,
- then lower `--runtime-option num_batch=24` to `16`,
- then try the `vram_constrained` profile again.

If that still fails, the model is probably too large for the current settings.

### Slow First Token

If the prompt starts slowly but eventually works:

- that is usually the model loading,
- not a broken setup,
- and the second run is often faster than the first.

To reduce the pain:

- keep the context modest,
- avoid huge prompts at first,
- and use a quantized model tag instead of a full-precision one.

### Fallback Profile Selection

Use this ladder:

1. Start with `vram_constrained`.
2. If the model still fails, lower context from `4096` to `2048`.
3. If that still fails, lower batch size from `24` to `16`.
4. If the GPU path is still unstable, lower `num_gpu` and keep `vram_constrained`.

The rule is simple: lower context first, lower batch second, lower offload third.

## Practical Advice

- For 14B+ models, start with a quantized tag.
- Do not begin with a huge context window unless you really need it.
- If you only want to test the plumbing, keep `--max-tokens` small.
- If the model is stuck on the first run, check the daemon and the backend tag before changing anything else.

## Phase 5 In One Sentence

Phase 5 is the step where AstraWeave starts behaving like a memory-aware control plane for big local models, instead of just a bridge to a backend.
