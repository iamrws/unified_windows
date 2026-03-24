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
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_RUNTIME_BACKEND = "ollama"
DEFAULT_PROMPT = "Explain in one paragraph why 128 GB of system RAM helps local LLM inference."
DEFAULT_STEP_NAME = "prompt_smoke"
DEFAULT_SERVICE_RUNSTEP_MODE = "simulation"
DEFAULT_IPC_TIMEOUT_SECONDS = float(os.environ.get("ASTRAWEAVE_SMOKE_IPC_TIMEOUT_SECONDS", "120"))


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
    ollama_base_url: str = DEFAULT_OLLAMA_URL,
    temperature: float = 0.2,
    max_tokens: int = 128,
    service_runstep_mode: str = DEFAULT_SERVICE_RUNSTEP_MODE,
    ipc_timeout_seconds: float = DEFAULT_IPC_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    if runtime_backend == "ollama":
        # Service-side Ollama runtime reads this env var while loading backend adapters.
        os.environ["ASTRAWEAVE_OLLAMA_BASE_URL"] = ollama_base_url.rstrip("/")

    from astrawave.ipc_client import AstraWeaveIpcClient
    from astrawave.ipc_server import AstraWeaveIpcServer
    from astrawave.service import AstraWeaveService

    if not model_name.strip():
        raise ValueError("model_name must not be empty")
    if not prompt.strip():
        raise ValueError("prompt must not be empty")

    service = AstraWeaveService(runstep_mode=service_runstep_mode)
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
            client.LoadModel(session_id, model_name, runtime_backend=runtime_backend)
            run_step_result = client.RunStep(
                session_id,
                step_name=DEFAULT_STEP_NAME,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
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

        return {
            "run_id": started_at,
            "generated_at": ended_at,
            "model_name": model_name,
            "runtime_backend": runtime_backend,
            "prompt_sha256": prompt_hash,
            "prompt_length": len(prompt),
            "service": {
                "endpoint": endpoint,
                "caller": _json_safe(caller),
                "session_id": session_id,
                "load_model": {
                    "session_id": session_id,
                    "model_name": model_name,
                    "runtime_backend": runtime_backend,
                },
                "run_step": _json_safe(run_step_result),
            },
            "inference": {
                "provider": provider,
                "base_url": ollama_base_url.rstrip("/") if provider == "ollama" else None,
                "model_name": result_model,
                "temperature": temperature,
                "max_tokens": max_tokens,
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
