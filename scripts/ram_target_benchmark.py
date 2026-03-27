"""RAM-target benchmark and tuning sweep for local Ollama inference.

This script helps operators target very high host-RAM usage (for example 90 GB)
while keeping the service path reproducible and measurable.

It executes multiple live-inference smoke runs with compression-aware runtime
option candidates, samples Ollama process memory, and scores the candidates by:
1) closeness to the RAM target and
2) decode throughput (tokens/sec).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from time import sleep
from typing import Any, Mapping, Sequence
from urllib import error, request

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from astrawave.runtime_tuning import infer_model_size_billion
from scripts.live_inference_smoke import run_live_inference_smoke

DEFAULT_TARGET_RAM_GB = 90.0
DEFAULT_RUNTIME_PROFILE = "vram_constrained"
DEFAULT_RUNTIME_BACKEND = "ollama"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_PROMPT = "Explain in simple terms how host RAM is being used for this inference run."
DEFAULT_MAX_TOKENS = 128
DEFAULT_TEMPERATURE = 0.2
DEFAULT_IPC_TIMEOUT_SECONDS = 900.0
DEFAULT_COOLDOWN_SECONDS = 2.0
DEFAULT_ITERATIONS = 2


@dataclass(frozen=True, slots=True)
class CandidateConfig:
    candidate_id: str
    compression_hint: str
    notes: str
    runtime_options: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CandidateRun:
    candidate_id: str
    iteration: int
    ok: bool
    error: str | None
    memory_before: dict[str, Any]
    memory_after: dict[str, Any]
    ollama_ps_before: dict[str, Any]
    ollama_ps_after: dict[str, Any]
    metrics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CandidateSummary:
    candidate_id: str
    compression_hint: str
    notes: str
    runtime_options: dict[str, Any]
    runs: list[dict[str, Any]]
    success_count: int
    run_count: int
    success_rate: float
    avg_eval_tokens_per_second: float | None
    avg_end_to_end_tokens_per_second: float | None
    avg_total_seconds: float | None
    peak_private_gb: float | None
    peak_working_set_gb: float | None
    ram_target_delta_gb: float | None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _to_gb(value_bytes: int | float | None) -> float | None:
    if value_bytes is None:
        return None
    return float(value_bytes) / float(1024**3)


def _ns_to_seconds(value_ns: Any) -> float | None:
    numeric = _safe_float(value_ns)
    if numeric is None:
        return None
    return numeric / 1_000_000_000.0


def _extract_metrics(smoke_result: Mapping[str, Any]) -> dict[str, Any]:
    inference = smoke_result.get("inference")
    if not isinstance(inference, Mapping):
        return {"error": "missing inference envelope"}
    result = inference.get("result")
    if not isinstance(result, Mapping):
        return {"error": "missing inference result"}
    usage = result.get("usage")
    if not isinstance(usage, Mapping):
        usage = {}
    raw = result.get("raw")
    if not isinstance(raw, Mapping):
        raw = {}

    eval_count = _safe_float(usage.get("eval_count", raw.get("eval_count")))
    eval_duration_s = _ns_to_seconds(usage.get("eval_duration", raw.get("eval_duration")))
    total_duration_s = _ns_to_seconds(usage.get("total_duration", raw.get("total_duration")))
    load_duration_s = _ns_to_seconds(usage.get("load_duration", raw.get("load_duration")))
    prompt_eval_count = _safe_float(usage.get("prompt_eval_count", raw.get("prompt_eval_count")))
    prompt_eval_duration_s = _ns_to_seconds(usage.get("prompt_eval_duration", raw.get("prompt_eval_duration")))

    eval_tps = None
    if eval_count is not None and eval_duration_s is not None and eval_duration_s > 0:
        eval_tps = eval_count / eval_duration_s

    end_to_end_tps = None
    if eval_count is not None and total_duration_s is not None and total_duration_s > 0:
        end_to_end_tps = eval_count / total_duration_s

    return {
        "eval_count": eval_count,
        "eval_duration_seconds": eval_duration_s,
        "total_duration_seconds": total_duration_s,
        "load_duration_seconds": load_duration_s,
        "prompt_eval_count": prompt_eval_count,
        "prompt_eval_duration_seconds": prompt_eval_duration_s,
        "eval_tokens_per_second": eval_tps,
        "end_to_end_tokens_per_second": end_to_end_tps,
        "finish_reason": result.get("finish_reason"),
    }


_ALLOWED_POWERSHELL_COMMANDS: frozenset[str] = frozenset({
    "Get-Process",
    "Select-Object",
    "Sort-Object",
    "ConvertTo-Json",
    "Where-Object",
})


def _run_powershell(command: str) -> str:
    # H26 fix: validate that the command starts with an allowed PowerShell cmdlet
    # to prevent command-injection via arbitrary strings.
    stripped = command.lstrip("$")
    first_token = stripped.split()[0] if stripped.split() else ""
    # The command may start with a variable assignment like "$p = Get-Process ..."
    # so also check after the first "=" or after "$var = ".
    tokens_to_check = [first_token]
    if "=" in command:
        after_eq = command.split("=", 1)[1].strip()
        if after_eq:
            tokens_to_check.append(after_eq.split()[0])
    if not any(token in _ALLOWED_POWERSHELL_COMMANDS for token in tokens_to_check):
        raise ValueError(
            f"PowerShell command not in allowlist. "
            f"First tokens: {tokens_to_check!r}, allowed: {sorted(_ALLOWED_POWERSHELL_COMMANDS)}"
        )
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"PowerShell command failed: {command}")
    return completed.stdout.strip()


def _collect_ollama_process_memory() -> dict[str, Any]:
    cmd = (
        "$p = Get-Process ollama -ErrorAction SilentlyContinue | "
        "Sort-Object WorkingSet64 -Descending | Select-Object -First 1 "
        "Id,ProcessName,WorkingSet64,PrivateMemorySize64; "
        "if ($null -eq $p) { '{}' } else { $p | ConvertTo-Json -Compress }"
    )
    try:
        text = _run_powershell(cmd)
    except Exception as exc:
        return {"observed": False, "error": str(exc), "source": "powershell"}
    if not text:
        return {"observed": False, "source": "powershell"}
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        return {"observed": False, "error": "failed to parse PowerShell JSON output", "source": "powershell"}
    if not isinstance(decoded, Mapping) or not decoded:
        return {"observed": False, "source": "powershell"}

    working_set_bytes = decoded.get("WorkingSet64")
    private_bytes = decoded.get("PrivateMemorySize64")
    ws_float = _safe_float(working_set_bytes)
    private_float = _safe_float(private_bytes)
    return {
        "observed": True,
        "source": "powershell",
        "process_id": decoded.get("Id"),
        "process_name": decoded.get("ProcessName"),
        "working_set_bytes": int(ws_float) if ws_float is not None else None,
        "private_bytes": int(private_float) if private_float is not None else None,
        "working_set_gb": _to_gb(ws_float),
        "private_gb": _to_gb(private_float),
    }


def _collect_ollama_ps(base_url: str, timeout_seconds: float = 5.0) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/api/ps"
    req = request.Request(url=url, method="GET")
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except error.URLError as exc:
        return {"observed": False, "error": str(exc.reason), "source": "ollama_ps"}
    except Exception as exc:  # pragma: no cover - defensive boundary
        return {"observed": False, "error": str(exc), "source": "ollama_ps"}

    try:
        decoded = json.loads(body)
    except json.JSONDecodeError:
        return {"observed": False, "error": "invalid JSON from /api/ps", "source": "ollama_ps"}
    if not isinstance(decoded, Mapping):
        return {"observed": False, "error": "unexpected /api/ps payload", "source": "ollama_ps"}

    models = decoded.get("models")
    if not isinstance(models, list):
        models = []
    top_size_bytes: int | None = None
    top_size_vram_bytes: int | None = None
    if models:
        candidate_models = [item for item in models if isinstance(item, Mapping)]
        if candidate_models:
            by_size = sorted(
                candidate_models,
                key=lambda item: float(item.get("size", 0) or 0),
                reverse=True,
            )
            top = by_size[0]
            top_size = _safe_float(top.get("size"))
            top_size_vram = _safe_float(top.get("size_vram"))
            top_size_bytes = int(top_size) if top_size is not None else None
            top_size_vram_bytes = int(top_size_vram) if top_size_vram is not None else None

    return {
        "observed": True,
        "source": "ollama_ps",
        "model_count": len(models),
        "models": decoded.get("models"),
        "top_size_bytes": top_size_bytes,
        "top_size_vram_bytes": top_size_vram_bytes,
        "top_size_gb": _to_gb(top_size_bytes),
        "top_size_vram_gb": _to_gb(top_size_vram_bytes),
    }


def build_default_candidates(model_size_billion: float | None) -> list[CandidateConfig]:
    """Return compression-aware candidate options.

    These are option profiles, not quantization guarantees. The quantization
    still depends on which model tag the operator has pulled.
    """

    if model_size_billion is None:
        size_class = "unknown"
    elif model_size_billion >= 60.0:
        size_class = "70b_class"
    elif model_size_billion >= 30.0:
        size_class = "30b_class"
    else:
        size_class = "14b_class"

    if size_class == "70b_class":
        return [
            CandidateConfig(
                candidate_id="q4_fit",
                compression_hint="Prefer Q4_K_M or equivalent",
                notes="Safest fit-first profile for very large models.",
                runtime_options={"num_ctx": 2048, "num_batch": 12, "num_gpu": 18, "top_p": 0.9, "repeat_penalty": 1.1},
            ),
            CandidateConfig(
                candidate_id="q4_balance",
                compression_hint="Prefer Q4_K_M or equivalent",
                notes="Moderate context and throughput while preserving fit margin.",
                runtime_options={"num_ctx": 3072, "num_batch": 16, "num_gpu": 20, "top_p": 0.9, "repeat_penalty": 1.1},
            ),
            CandidateConfig(
                candidate_id="q6_push",
                compression_hint="Prefer Q6_K if memory allows",
                notes="Higher quality target; may exceed fit on tight systems.",
                runtime_options={"num_ctx": 2048, "num_batch": 8, "num_gpu": 20, "top_p": 0.92, "repeat_penalty": 1.08},
            ),
        ]

    if size_class == "30b_class":
        return [
            CandidateConfig(
                candidate_id="q4_balance",
                compression_hint="Prefer Q4_K_M or equivalent",
                notes="Balanced profile for 30B class models.",
                runtime_options={"num_ctx": 4096, "num_batch": 20, "num_gpu": 20, "top_p": 0.9, "repeat_penalty": 1.1},
            ),
            CandidateConfig(
                candidate_id="q6_balance",
                compression_hint="Prefer Q6_K if memory allows",
                notes="More quality, slightly heavier memory profile.",
                runtime_options={"num_ctx": 3072, "num_batch": 16, "num_gpu": 22, "top_p": 0.92, "repeat_penalty": 1.08},
            ),
            CandidateConfig(
                candidate_id="q8_push",
                compression_hint="Prefer Q8_0 only when fit margin is large",
                notes="Aggressive memory push for high-RAM hosts.",
                runtime_options={"num_ctx": 4096, "num_batch": 24, "num_gpu": 24, "top_p": 0.95, "repeat_penalty": 1.05},
            ),
        ]

    return [
        CandidateConfig(
            candidate_id="q4_balance",
            compression_hint="Prefer Q4_K_M or equivalent",
            notes="Default profile for 14B class models.",
            runtime_options={"num_ctx": 4096, "num_batch": 24, "num_gpu": 24, "top_p": 0.9, "repeat_penalty": 1.1},
        ),
        CandidateConfig(
            candidate_id="q4_context_push",
            compression_hint="Prefer Q4_K_M or equivalent",
            notes="Higher context for RAM-rich hosts; slower decode expected.",
            runtime_options={"num_ctx": 6144, "num_batch": 20, "num_gpu": 24, "top_p": 0.9, "repeat_penalty": 1.1},
        ),
        CandidateConfig(
            candidate_id="q6_balance",
            compression_hint="Prefer Q6_K if memory allows",
            notes="Quality-oriented profile with tighter context.",
            runtime_options={"num_ctx": 3072, "num_batch": 16, "num_gpu": 24, "top_p": 0.92, "repeat_penalty": 1.08},
        ),
    ]


def _distance_to_target_bytes(summary: CandidateSummary, target_ram_bytes: float) -> float:
    peak_private_gb = summary.peak_private_gb
    peak_working_set_gb = summary.peak_working_set_gb
    observed_gb = peak_private_gb if peak_private_gb is not None else peak_working_set_gb
    if observed_gb is None:
        return float("inf")
    return abs((observed_gb * float(1024**3)) - target_ram_bytes)


def select_best_candidate(
    summaries: Sequence[CandidateSummary],
    *,
    target_ram_gb: float,
) -> CandidateSummary | None:
    target_ram_bytes = float(target_ram_gb) * float(1024**3)
    successful = [item for item in summaries if item.success_count > 0]
    if not successful:
        return None
    return min(
        successful,
        key=lambda item: (
            _distance_to_target_bytes(item, target_ram_bytes),
            -(item.avg_eval_tokens_per_second or 0.0),
            -(item.avg_end_to_end_tokens_per_second or 0.0),
        ),
    )


def _summarize_runs(
    candidate: CandidateConfig,
    runs: Sequence[CandidateRun],
    *,
    target_ram_gb: float,
) -> CandidateSummary:
    run_dicts = [asdict(item) for item in runs]
    success_runs = [item for item in runs if item.ok]

    def _mean(values: list[float]) -> float | None:
        if not values:
            return None
        return sum(values) / len(values)

    eval_tps_values = [
        float(item.metrics.get("eval_tokens_per_second"))
        for item in success_runs
        if _safe_float(item.metrics.get("eval_tokens_per_second")) is not None
    ]
    end_to_end_tps_values = [
        float(item.metrics.get("end_to_end_tokens_per_second"))
        for item in success_runs
        if _safe_float(item.metrics.get("end_to_end_tokens_per_second")) is not None
    ]
    total_seconds_values = [
        float(item.metrics.get("total_duration_seconds"))
        for item in success_runs
        if _safe_float(item.metrics.get("total_duration_seconds")) is not None
    ]

    peak_private_gb = None
    peak_working_set_gb = None
    for item in success_runs:
        private_gb = _safe_float(item.memory_after.get("private_gb"))
        ws_gb = _safe_float(item.memory_after.get("working_set_gb"))
        if private_gb is not None:
            peak_private_gb = private_gb if peak_private_gb is None else max(peak_private_gb, private_gb)
        if ws_gb is not None:
            peak_working_set_gb = ws_gb if peak_working_set_gb is None else max(peak_working_set_gb, ws_gb)

    observed_gb = peak_private_gb if peak_private_gb is not None else peak_working_set_gb
    ram_target_delta_gb = None
    if observed_gb is not None:
        ram_target_delta_gb = observed_gb - float(target_ram_gb)

    run_count = len(runs)
    success_count = len(success_runs)
    success_rate = (float(success_count) / float(run_count)) if run_count > 0 else 0.0

    return CandidateSummary(
        candidate_id=candidate.candidate_id,
        compression_hint=candidate.compression_hint,
        notes=candidate.notes,
        runtime_options=dict(candidate.runtime_options),
        runs=run_dicts,
        success_count=success_count,
        run_count=run_count,
        success_rate=success_rate,
        avg_eval_tokens_per_second=_mean(eval_tps_values),
        avg_end_to_end_tokens_per_second=_mean(end_to_end_tps_values),
        avg_total_seconds=_mean(total_seconds_values),
        peak_private_gb=peak_private_gb,
        peak_working_set_gb=peak_working_set_gb,
        ram_target_delta_gb=ram_target_delta_gb,
    )


def _default_output_paths(run_id: str) -> tuple[Path, Path]:
    target_dir = REPO_ROOT / "reports" / "benchmarks"
    target_dir.mkdir(parents=True, exist_ok=True)
    base = target_dir / f"ram_target_{run_id}"
    return base.with_suffix(".json"), base.with_suffix(".md")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# RAM Target Benchmark Report")
    lines.append("")
    lines.append(f"- Generated at: `{payload.get('generated_at')}`")
    lines.append(f"- Run ID: `{payload.get('run_id')}`")
    lines.append(f"- Model: `{payload.get('model_name')}`")
    lines.append(f"- Target RAM (GB): `{payload.get('target_ram_gb')}`")
    lines.append("")

    best = payload.get("best_candidate")
    if isinstance(best, Mapping):
        lines.append("## Recommended Candidate")
        lines.append("")
        lines.append(f"- Candidate ID: `{best.get('candidate_id')}`")
        lines.append(f"- Compression Hint: `{best.get('compression_hint')}`")
        lines.append(f"- Success Rate: `{best.get('success_rate')}`")
        lines.append(f"- Avg Eval Tokens/Sec: `{best.get('avg_eval_tokens_per_second')}`")
        lines.append(f"- Peak Private GB: `{best.get('peak_private_gb')}`")
        lines.append(f"- Peak Working Set GB: `{best.get('peak_working_set_gb')}`")
        lines.append(f"- RAM Delta GB (observed - target): `{best.get('ram_target_delta_gb')}`")
        lines.append(f"- Runtime Options: `{best.get('runtime_options')}`")
        lines.append("")

    summaries = payload.get("candidate_summaries")
    if isinstance(summaries, list):
        lines.append("## All Candidates")
        lines.append("")
        for item in summaries:
            if not isinstance(item, Mapping):
                continue
            lines.append(f"### `{item.get('candidate_id')}`")
            lines.append(f"- Compression Hint: `{item.get('compression_hint')}`")
            lines.append(f"- Success Rate: `{item.get('success_rate')}`")
            lines.append(f"- Avg Eval Tokens/Sec: `{item.get('avg_eval_tokens_per_second')}`")
            lines.append(f"- Peak Private GB: `{item.get('peak_private_gb')}`")
            lines.append(f"- Peak Working Set GB: `{item.get('peak_working_set_gb')}`")
            lines.append(f"- RAM Delta GB: `{item.get('ram_target_delta_gb')}`")
            lines.append(f"- Runtime Options: `{item.get('runtime_options')}`")
            lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _run_candidate_iteration(
    *,
    candidate: CandidateConfig,
    iteration: int,
    model_name: str,
    prompt: str,
    runtime_backend: str,
    runtime_profile: str,
    ollama_base_url: str,
    temperature: float,
    max_tokens: int,
    service_runstep_mode: str,
    ipc_timeout_seconds: float,
) -> CandidateRun:
    memory_before = _collect_ollama_process_memory()
    ollama_ps_before = _collect_ollama_ps(ollama_base_url)
    try:
        smoke_result = run_live_inference_smoke(
            model_name=model_name,
            prompt=prompt,
            runtime_backend=runtime_backend,
            runtime_profile=runtime_profile,
            runtime_backend_options=candidate.runtime_options,
            runtime_profile_override=None,
            runtime_backend_options_override=None,
            ollama_base_url=ollama_base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            service_runstep_mode=service_runstep_mode,
            ipc_timeout_seconds=ipc_timeout_seconds,
        )
        metrics = _extract_metrics(smoke_result)
        ok = True
        error_text = None
    except Exception as exc:
        metrics = {"error": str(exc)}
        ok = False
        error_text = str(exc)

    memory_after = _collect_ollama_process_memory()
    ollama_ps_after = _collect_ollama_ps(ollama_base_url)

    return CandidateRun(
        candidate_id=candidate.candidate_id,
        iteration=iteration,
        ok=ok,
        error=error_text,
        memory_before=memory_before,
        memory_after=memory_after,
        ollama_ps_before=ollama_ps_before,
        ollama_ps_after=ollama_ps_after,
        metrics=metrics,
    )


def run_ram_target_benchmark(
    *,
    model_name: str,
    target_ram_gb: float,
    runtime_profile: str,
    runtime_backend: str,
    prompt: str,
    ollama_base_url: str,
    temperature: float,
    max_tokens: int,
    service_runstep_mode: str,
    ipc_timeout_seconds: float,
    iterations: int,
    cooldown_seconds: float,
) -> dict[str, Any]:
    model_size_billion = infer_model_size_billion(model_name)
    candidates = build_default_candidates(model_size_billion)
    summaries: list[CandidateSummary] = []

    for candidate in candidates:
        runs: list[CandidateRun] = []
        for iteration in range(1, iterations + 1):
            run = _run_candidate_iteration(
                candidate=candidate,
                iteration=iteration,
                model_name=model_name,
                prompt=prompt,
                runtime_backend=runtime_backend,
                runtime_profile=runtime_profile,
                ollama_base_url=ollama_base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                service_runstep_mode=service_runstep_mode,
                ipc_timeout_seconds=ipc_timeout_seconds,
            )
            runs.append(run)
            if cooldown_seconds > 0 and iteration < iterations:
                sleep(cooldown_seconds)
        summaries.append(_summarize_runs(candidate, runs, target_ram_gb=target_ram_gb))

    best = select_best_candidate(summaries, target_ram_gb=target_ram_gb)
    report: dict[str, Any] = {
        "generated_at": _utc_now_iso(),
        "model_name": model_name,
        "model_size_billion": model_size_billion,
        "target_ram_gb": target_ram_gb,
        "runtime_profile": runtime_profile,
        "runtime_backend": runtime_backend,
        "prompt_length": len(prompt),
        "iterations_per_candidate": iterations,
        "cooldown_seconds": cooldown_seconds,
        "candidate_summaries": [asdict(item) for item in summaries],
        "best_candidate": asdict(best) if best is not None else None,
    }
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a compression-aware RAM-target benchmark sweep for local Ollama inference."
    )
    parser.add_argument("--model", required=True, help="Model tag to benchmark (for example: qwen2.5:14b).")
    parser.add_argument("--target-ram-gb", type=float, default=DEFAULT_TARGET_RAM_GB, help="Target host RAM usage in GB.")
    parser.add_argument("--runtime-profile", default=DEFAULT_RUNTIME_PROFILE, help="Runtime profile to use for all candidates.")
    parser.add_argument("--runtime-backend", default=DEFAULT_RUNTIME_BACKEND, choices=["ollama", "simulation"])
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt used for each benchmark run.")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Ollama base URL.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--service-runstep-mode", choices=["simulation", "auto", "hardware"], default="simulation")
    parser.add_argument("--ipc-timeout-seconds", type=float, default=DEFAULT_IPC_TIMEOUT_SECONDS)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS, help="Runs per candidate option profile.")
    parser.add_argument("--cooldown-seconds", type=float, default=DEFAULT_COOLDOWN_SECONDS, help="Sleep between iterations.")
    parser.add_argument("--run-id", default=None, help="Optional run identifier for report filenames.")
    parser.add_argument("--output-json", default=None, help="Optional explicit JSON report path.")
    parser.add_argument("--output-md", default=None, help="Optional explicit Markdown report path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    report = run_ram_target_benchmark(
        model_name=args.model,
        target_ram_gb=float(args.target_ram_gb),
        runtime_profile=args.runtime_profile,
        runtime_backend=args.runtime_backend,
        prompt=args.prompt,
        ollama_base_url=args.ollama_url,
        temperature=float(args.temperature),
        max_tokens=int(args.max_tokens),
        service_runstep_mode=args.service_runstep_mode,
        ipc_timeout_seconds=float(args.ipc_timeout_seconds),
        iterations=max(int(args.iterations), 1),
        cooldown_seconds=max(float(args.cooldown_seconds), 0.0),
    )
    report["run_id"] = run_id

    default_json_path, default_md_path = _default_output_paths(run_id)
    json_path = Path(args.output_json) if args.output_json else default_json_path
    md_path = Path(args.output_md) if args.output_md else default_md_path

    _write_json(json_path, report)
    _write_markdown(md_path, report)

    output = {
        "ok": True,
        "result": {
            "run_id": run_id,
            "json_report": str(json_path),
            "markdown_report": str(md_path),
            "best_candidate": report.get("best_candidate"),
        },
    }
    print(json.dumps(output, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
