"""Smoke test for local AstraWeave session orchestration plus local prompt inference.

This script is intentionally honest about the current split:
- AstraWeave owns session creation, model registration, and orchestration.
- A local backend such as Ollama handles prompt generation.

It is a bridge workflow for testing a Windows host with large system RAM and a
small dGPU while service-native prompt routing is still landing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib import error, request

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from astrawave.inference_runtime import InferenceModelBinding, SimulationInferenceRuntime

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_RUNTIME_BACKEND = "ollama"
DEFAULT_PROMPT = "Explain in one paragraph why 128 GB of system RAM helps local LLM inference."
DEFAULT_STEP_NAME = "prompt_smoke"
DEFAULT_SERVICE_RUNSTEP_MODE = "simulation"
DEFAULT_IPC_TIMEOUT_SECONDS = float(os.environ.get("ASTRAWEAVE_SMOKE_IPC_TIMEOUT_SECONDS", "120"))
_POSITIVE_INT_OPTION_KEYS = {"num_ctx", "num_batch", "num_predict", "num_keep", "top_k"}
_NONNEGATIVE_INT_OPTION_KEYS = {"gpu_layers", "num_gpu"}
_NONNEGATIVE_FLOAT_OPTION_KEYS = {"temperature"}
_OPEN_INTERVAL_FLOAT_OPTION_KEYS = {"top_p", "repeat_penalty"}
_PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "balanced": {
        "num_ctx": 8192,
        "num_batch": 2,
        "num_keep": 128,
        "top_p": 0.95,
        "repeat_penalty": 1.1,
    },
    "vram_constrained": {
        "num_ctx": 4096,
        "num_batch": 1,
        "num_keep": 64,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
    },
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_safe(item) for item in value]
    if hasattr(value, "value"):
        return getattr(value, "value")
    return value


def _write_json(payload: Mapping[str, Any], *, stream: str = "stdout") -> None:
    text = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True)
    if stream == "stderr":
        print(text, file=sys.stderr)
    else:
        print(text)


def _service_endpoint_uri(endpoint: Any) -> str:
    if isinstance(endpoint, tuple) and len(endpoint) == 2:
        host, port = endpoint
        return f"tcp://{host}:{port}"
    if isinstance(endpoint, str):
        return endpoint
    return str(endpoint)


def _normalize_profile(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("runtime profile must be a non-empty string")
    return value.strip()


def _parse_model_size_billion(model_name: str) -> float | None:
    matches = re.findall(r"(?<!\d)(\d+(?:\.\d+)?)\s*[bB](?!\w)", model_name)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _resolve_runtime_profile(model_name: str, runtime_profile: str | None) -> tuple[str, float | None]:
    parsed_size = _parse_model_size_billion(model_name)
    normalized_profile = _normalize_profile(runtime_profile)
    if normalized_profile is not None and normalized_profile != "auto":
        return normalized_profile, parsed_size
    if parsed_size is not None and parsed_size >= 14.0:
        return "vram_constrained", parsed_size
    return "balanced", parsed_size


def _parse_option_pairs(
    raw_options: Mapping[str, Any] | None,
    *,
    field_name: str,
) -> dict[str, Any] | None:
    if raw_options is None:
        return None
    if not isinstance(raw_options, Mapping):
        raise ValueError(f"{field_name} must be a mapping")

    normalized: dict[str, Any] = {}
    for key, value in raw_options.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{field_name} keys must be non-empty strings")
        normalized[key.strip()] = _validate_option_value(field_name, key.strip(), value)
    return normalized


def _parse_cli_option_pairs(raw_options: Sequence[str] | None, *, field_name: str) -> dict[str, Any] | None:
    if raw_options is None:
        return None
    parsed: dict[str, Any] = {}
    for raw_item in raw_options:
        if "=" not in raw_item:
            raise ValueError(f"{field_name} entries must use key=value syntax")
        key_text, raw_value = raw_item.split("=", 1)
        key = key_text.strip()
        if not key:
            raise ValueError(f"{field_name} keys must be non-empty strings")
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        parsed[key] = _validate_option_value(field_name, key, value)
    return parsed


def _validate_option_value(field_name: str, key: str, value: Any) -> Any:
    if key in _POSITIVE_INT_OPTION_KEYS:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{field_name}.{key} must be a positive integer")
        return value
    if key in _NONNEGATIVE_INT_OPTION_KEYS:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{field_name}.{key} must be a non-negative integer")
        return value
    if key in _NONNEGATIVE_FLOAT_OPTION_KEYS:
        if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) < 0.0:
            raise ValueError(f"{field_name}.{key} must be a non-negative number")
        return float(value)
    if key == "top_p":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{field_name}.{key} must be a number")
        numeric_value = float(value)
        if numeric_value <= 0.0 or numeric_value > 1.0:
            raise ValueError(f"{field_name}.{key} must be between 0 and 1")
        return numeric_value
    if key in _OPEN_INTERVAL_FLOAT_OPTION_KEYS:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{field_name}.{key} must be a number")
        numeric_value = float(value)
        if numeric_value <= 0.0:
            raise ValueError(f"{field_name}.{key} must be greater than 0")
        return numeric_value
    return value


def _merge_runtime_options(*layers: Mapping[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for layer in layers:
        if layer is None:
            continue
        for key, value in layer.items():
            merged[key] = value
    return merged


def _synthesize_load_options(profile: str, model_size_billion: float | None, requested_options: Mapping[str, Any] | None) -> dict[str, Any]:
    options = dict(_PROFILE_DEFAULTS.get(profile, _PROFILE_DEFAULTS["balanced"]))
    if model_size_billion is not None and model_size_billion >= 32.0:
        options["num_ctx"] = min(int(options["num_ctx"]), 3072)
        options["num_batch"] = 1
    return _merge_runtime_options(options, requested_options)


def _synthesize_step_options(
    load_options: Mapping[str, Any],
    requested_options: Mapping[str, Any] | None,
    *,
    max_tokens: int | None,
    temperature: float | None,
) -> dict[str, Any]:
    options = dict(load_options)
    if max_tokens is not None and "num_predict" not in options:
        options["num_predict"] = max_tokens
    if temperature is not None and "temperature" not in options:
        options["temperature"] = temperature
    return _merge_runtime_options(options, requested_options)


def _post_json(url: str, payload: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_seconds) as response:
        body = response.read().decode("utf-8")
    decoded = json.loads(body)
    if not isinstance(decoded, dict):
        raise RuntimeError("ollama response must be a JSON object")
    return decoded


class _ProfiledOllamaRuntime:
    backend_name = "ollama"

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        tuning_state: dict[str, Any],
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._tuning_state = tuning_state

    def load_model(self, model_name: str) -> InferenceModelBinding:
        requested_profile = self._tuning_state.get("load_requested_profile")
        requested_options = _parse_option_pairs(
            self._tuning_state.get("load_requested_options"),
            field_name="runtime_backend_options",
        )
        effective_profile, model_size_billion = _resolve_runtime_profile(model_name, requested_profile)
        backend_options = _synthesize_load_options(effective_profile, model_size_billion, requested_options)
        self._tuning_state["effective_load_profile"] = effective_profile
        self._tuning_state["model_size_billion"] = model_size_billion
        self._tuning_state["effective_load_options"] = backend_options
        return InferenceModelBinding(
            backend=self.backend_name,
            requested_model_name=model_name,
            resolved_model_name=model_name,
            metadata={
                "base_url": self._base_url,
                "model_size_billion": model_size_billion,
                "runtime_profile": effective_profile,
                "runtime_backend_options": backend_options,
                "supports_prompt_generation": True,
                "transport": "http",
            },
        )

    def generate(
        self,
        model_name: str,
        *,
        prompt: str,
        step_name: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        step_profile = _normalize_profile(self._tuning_state.get("step_requested_profile"))
        step_requested_options = _parse_option_pairs(
            self._tuning_state.get("step_requested_options"),
            field_name="runtime_backend_options_override",
        )
        load_profile = self._tuning_state.get("effective_load_profile")
        effective_profile = step_profile or load_profile or "balanced"
        load_options = self._tuning_state.get("effective_load_options")
        if not isinstance(load_options, Mapping):
            load_options = {}
        backend_options = _synthesize_step_options(
            load_options,
            step_requested_options,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self._tuning_state["effective_step_profile"] = effective_profile
        self._tuning_state["effective_step_options"] = backend_options
        payload: dict[str, Any] = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
        }
        if system_prompt:
            payload["system"] = system_prompt
        options = dict(backend_options)
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if temperature is not None and "temperature" not in options:
            options["temperature"] = temperature
        if options:
            payload["options"] = options
        raw = _post_json(f"{self._base_url}/api/generate", payload, self._timeout_seconds)
        output_text = str(raw.get("response", ""))
        return {
            "ok": True,
            "backend": self.backend_name,
            "model_name": model_name,
            "output_text": output_text,
            "finish_reason": raw.get("done_reason") or ("stop" if raw.get("done") else "unknown"),
            "usage": {
                "prompt_eval_count": raw.get("prompt_eval_count"),
                "eval_count": raw.get("eval_count"),
                "total_duration": raw.get("total_duration"),
                "load_duration": raw.get("load_duration"),
                "eval_duration": raw.get("eval_duration"),
                "step_name": step_name,
            },
            "raw": {key: value for key, value in raw.items() if key not in {"prompt", "context"}},
        }


def _current_caller() -> CallerIdentity:
    from astrawave.security import CallerIdentity, resolve_current_user_sid
    from astrawave.service import DEFAULT_SERVICE_OWNER_SID

    sid = resolve_current_user_sid() or DEFAULT_SERVICE_OWNER_SID
    return CallerIdentity(user_sid=sid, pid=os.getpid())


def run_live_inference_smoke(
    *,
    model_name: str,
    prompt: str,
    runtime_backend: str = DEFAULT_RUNTIME_BACKEND,
    runtime_profile: str | None = None,
    runtime_backend_options: Mapping[str, Any] | None = None,
    runtime_profile_override: str | None = None,
    runtime_backend_options_override: Mapping[str, Any] | None = None,
    ollama_base_url: str = DEFAULT_OLLAMA_URL,
    temperature: float = 0.2,
    max_tokens: int = 128,
    service_runstep_mode: str = DEFAULT_SERVICE_RUNSTEP_MODE,
    ipc_timeout_seconds: float = DEFAULT_IPC_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    # H27 fix: instead of mutating os.environ, the base_url is passed directly
    # to _ProfiledOllamaRuntime via the factory closure below.

    from astrawave.ipc_client import AstraWeaveIpcClient
    from astrawave.ipc_server import AstraWeaveIpcServer
    from astrawave.service import AstraWeaveService

    if not model_name.strip():
        raise ValueError("model_name must not be empty")
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    requested_load_profile = _normalize_profile(runtime_profile)
    requested_load_options = _parse_option_pairs(runtime_backend_options, field_name="runtime_backend_options")
    requested_step_profile = _normalize_profile(runtime_profile_override)
    requested_step_options = _parse_option_pairs(
        runtime_backend_options_override,
        field_name="runtime_backend_options_override",
    )
    planned_load_profile, model_size_billion = _resolve_runtime_profile(model_name, requested_load_profile)
    planned_load_options = _synthesize_load_options(planned_load_profile, model_size_billion, requested_load_options)
    planned_step_profile = requested_step_profile or planned_load_profile
    planned_step_options = _synthesize_step_options(
        planned_load_options,
        requested_step_options,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    tuning_state: dict[str, Any] = {
        "load_requested_profile": requested_load_profile,
        "load_requested_options": requested_load_options,
        "step_requested_profile": requested_step_profile,
        "step_requested_options": requested_step_options,
        "model_size_billion": model_size_billion,
        "effective_load_profile": planned_load_profile,
        "effective_load_options": planned_load_options,
        "effective_step_profile": planned_step_profile,
        "effective_step_options": planned_step_options,
    }

    def _inference_runtime_factory(backend_name: str):
        if backend_name == "simulation":
            return SimulationInferenceRuntime()
        if backend_name == "ollama":
            return _ProfiledOllamaRuntime(
                base_url=ollama_base_url,
                timeout_seconds=ipc_timeout_seconds,
                tuning_state=tuning_state,
            )
        raise ValueError(f"unsupported runtime backend: {backend_name}")

    service = AstraWeaveService(runstep_mode=service_runstep_mode, inference_runtime_factory=_inference_runtime_factory)
    server = AstraWeaveIpcServer(
        service=service,
        prefer_named_pipe=False,
        host="127.0.0.1",
        port=0,
    )
    caller = _current_caller()
    started_at = _utc_now_iso()

    server.start()
    try:
        endpoint = _service_endpoint_uri(server.endpoint)
        with AstraWeaveIpcClient(
            endpoint=endpoint,
            default_caller=caller,
            timeout=ipc_timeout_seconds,
        ) as client:
            session_id = client.CreateSession()
            client.LoadModel(
                session_id,
                model_name,
                runtime_backend=runtime_backend,
                runtime_profile=requested_load_profile,
                runtime_backend_options=requested_load_options,
            )
            run_step_result = client.RunStep(
                session_id,
                step_name=DEFAULT_STEP_NAME,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                runtime_profile_override=requested_step_profile,
                runtime_backend_options_override=requested_step_options,
            )
            client.CloseSession(session_id)

        inference_result = run_step_result.get("inference_result")
        if not isinstance(inference_result, Mapping):
            raise RuntimeError("service run-step did not return inference_result")
        if not bool(inference_result.get("ok", False)):
            error_block = inference_result.get("error")
            if isinstance(error_block, Mapping):
                code = error_block.get("code", "unknown")
                message = error_block.get("message", "unknown inference failure")
                raise RuntimeError(f"inference backend failed: {code} {message}")
            raise RuntimeError("inference backend failed without structured error details")

        ended_at = _utc_now_iso()
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        provider = str(inference_result.get("backend", runtime_backend))
        result_model = str(inference_result.get("model_name", model_name))
        effective_load_profile = tuning_state.get("effective_load_profile")
        if not isinstance(effective_load_profile, str):
            effective_load_profile = requested_load_profile
        effective_step_profile = tuning_state.get("effective_step_profile")
        if not isinstance(effective_step_profile, str):
            effective_step_profile = requested_step_profile or effective_load_profile
        effective_load_options = tuning_state.get("effective_load_options")
        if not isinstance(effective_load_options, Mapping):
            effective_load_options = requested_load_options or {}
        effective_step_options = tuning_state.get("effective_step_options")
        if not isinstance(effective_step_options, Mapping):
            effective_step_options = requested_step_options or {}
        run_step_report = _json_safe(run_step_result)
        if isinstance(run_step_report, dict):
            if requested_step_profile is not None:
                run_step_report["runtime_profile_override"] = requested_step_profile
            if requested_step_options is not None:
                run_step_report["runtime_backend_options_override"] = requested_step_options

        load_model_report: dict[str, Any] = {
            "session_id": session_id,
            "model_name": model_name,
            "runtime_backend": runtime_backend,
        }
        if requested_load_profile is not None:
            load_model_report["runtime_profile"] = requested_load_profile
        if requested_load_options is not None:
            load_model_report["runtime_backend_options"] = requested_load_options

        return {
            "run_id": started_at,
            "generated_at": ended_at,
            "model_name": model_name,
            "runtime_backend": runtime_backend,
            "prompt_sha256": prompt_hash,
            "prompt_length": len(prompt),
            "runtime_tuning": {
                "model_size_billion": tuning_state.get("model_size_billion"),
                "requested": {
                    "load_profile": requested_load_profile,
                    "load_backend_options": requested_load_options or {},
                    "step_profile_override": requested_step_profile,
                    "step_backend_options_override": requested_step_options or {},
                },
                "effective": {
                    "load_profile": effective_load_profile,
                    "load_backend_options": dict(effective_load_options),
                    "step_profile_override": effective_step_profile,
                    "step_backend_options_override": dict(effective_step_options),
                },
            },
            "service": {
                "endpoint": endpoint,
                "caller": _json_safe(caller),
                "session_id": session_id,
                "load_model": load_model_report,
                "run_step": run_step_report,
            },
            "inference": {
                "provider": provider,
                "base_url": ollama_base_url.rstrip("/") if provider == "ollama" else None,
                "model_name": result_model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "runtime_profile": effective_step_profile,
                "runtime_backend_options": dict(effective_step_options),
                "result": _json_safe(inference_result),
            },
        }
    finally:
        server.stop()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke test local AstraWeave orchestration plus local prompt inference.")
    parser.add_argument(
        "--model",
        required=True,
        help="Model identifier for the selected runtime backend.",
    )
    parser.add_argument(
        "--runtime-backend",
        choices=["ollama", "simulation"],
        default=DEFAULT_RUNTIME_BACKEND,
        help="Prompt runtime backend used by AstraWeave RunStep.",
    )
    parser.add_argument(
        "--runtime-profile",
        default=None,
        help="Optional load-time tuning profile (for example: auto or vram_constrained).",
    )
    parser.add_argument(
        "--runtime-option",
        dest="runtime_options",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Optional load-time backend tuning option; may be repeated.",
    )
    parser.add_argument(
        "--runtime-profile-override",
        default=None,
        help="Optional per-step tuning profile override.",
    )
    parser.add_argument(
        "--runtime-option-override",
        dest="runtime_option_overrides",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Optional per-step backend tuning option override; may be repeated.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt to send to the local inference backend.",
    )
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help="Base URL for the local Ollama server (used only when --runtime-backend=ollama).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature sent to the local backend.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum tokens requested from the local backend.",
    )
    parser.add_argument(
        "--service-runstep-mode",
        choices=["simulation", "auto", "hardware"],
        default=DEFAULT_SERVICE_RUNSTEP_MODE,
        help="RunStep mode used when the smoke script exercises AstraWeave orchestration.",
    )
    parser.add_argument(
        "--ipc-timeout-seconds",
        type=float,
        default=DEFAULT_IPC_TIMEOUT_SECONDS,
        help="IPC timeout used for service calls in this smoke script.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_live_inference_smoke(
            model_name=args.model,
            prompt=args.prompt,
            runtime_backend=args.runtime_backend,
            runtime_profile=args.runtime_profile,
            runtime_backend_options=_parse_cli_option_pairs(args.runtime_options, field_name="runtime_options"),
            runtime_profile_override=args.runtime_profile_override,
            runtime_backend_options_override=_parse_cli_option_pairs(
                args.runtime_option_overrides,
                field_name="runtime_option_overrides",
            ),
            ollama_base_url=args.ollama_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            service_runstep_mode=args.service_runstep_mode,
            ipc_timeout_seconds=args.ipc_timeout_seconds,
        )
        _write_json({"ok": True, "result": _json_safe(result)})
        return 0
    except Exception as exc:
        _write_json(
            {
                "ok": False,
                "error": {
                    "code": "SMOKE_FAILED",
                    "message": str(exc),
                },
            },
            stream="stderr",
        )
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
