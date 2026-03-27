"""Inference runtime adapters for local model execution."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
import ssl
from typing import Any, Callable, Mapping, Protocol
from urllib import error, request
from urllib.parse import urlparse

from .errors import ApiError, ApiErrorCode
from .runtime_tuning import merge_backend_options, resolve_runtime_tuning

# M14 fix: read env vars lazily via properties, not at import time
_DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
_DEFAULT_OLLAMA_TIMEOUT_SECONDS = 120.0


def _get_ollama_base_url() -> str:
    return os.environ.get("ASTRAWEAVE_OLLAMA_BASE_URL", _DEFAULT_OLLAMA_BASE_URL)


def _get_ollama_timeout() -> float:
    raw = os.environ.get("ASTRAWEAVE_OLLAMA_TIMEOUT_SECONDS")
    if raw is None:
        return _DEFAULT_OLLAMA_TIMEOUT_SECONDS
    try:
        return float(raw)
    except ValueError:
        return _DEFAULT_OLLAMA_TIMEOUT_SECONDS


# Keep module-level names for backward compat (read at access time)
DEFAULT_OLLAMA_BASE_URL = _DEFAULT_OLLAMA_BASE_URL
DEFAULT_OLLAMA_TIMEOUT_SECONDS = _DEFAULT_OLLAMA_TIMEOUT_SECONDS

InferenceTransport = Callable[[str, dict[str, Any], float], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class InferenceModelBinding:
    """Structured model binding returned by a runtime backend."""

    backend: str
    requested_model_name: str
    resolved_model_name: str
    metadata: dict[str, Any]


class InferenceRuntime(Protocol):
    """Minimal runtime adapter contract used by the service."""

    backend_name: str

    def load_model(
        self,
        model_name: str,
        *,
        runtime_profile: str | None = None,
        backend_options: Mapping[str, Any] | None = None,
    ) -> InferenceModelBinding:
        """Prepare or bind a model name for later generation."""

    def generate(
        self,
        model_name: str,
        *,
        prompt: str,
        step_name: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system_prompt: str | None = None,
        backend_options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate output for one step."""


class SimulationInferenceRuntime:
    """Deterministic fallback backend used when no real runtime is configured."""

    backend_name = "simulation"

    def load_model(
        self,
        model_name: str,
        *,
        runtime_profile: str | None = None,
        backend_options: Mapping[str, Any] | None = None,
    ) -> InferenceModelBinding:
        tuning = resolve_runtime_tuning(
            model_name,
            runtime_profile=runtime_profile,
            backend_options=backend_options,
        )
        return InferenceModelBinding(
            backend=self.backend_name,
            requested_model_name=model_name,
            resolved_model_name=model_name,
            metadata={
                "supports_prompt_generation": True,
                "transport": "in-process",
                "runtime_profile": tuning.profile_name,
                "model_size_billion": tuning.model_size_billion,
                "model_size_label": tuning.model_size_label,
                "backend_options": tuning.backend_options,
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
        backend_options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        del system_prompt
        output_text = (
            f"Simulated response from {model_name} for step '{step_name}' "
            f"using {len(prompt)} prompt chars."
        )
        return {
            "ok": True,
            "backend": self.backend_name,
            "model_name": model_name,
            "output_text": output_text,
            "finish_reason": "stop",
            "usage": {
                "prompt_chars": len(prompt),
                "completion_chars": len(output_text),
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        }


class OllamaInferenceRuntime:
    """HTTP adapter for a local Ollama daemon."""

    backend_name = "ollama"

    _LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
    _logger = logging.getLogger("astrawave.inference_runtime")

    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        transport: InferenceTransport | None = None,
    ) -> None:
        raw_url = (base_url or _get_ollama_base_url()).rstrip("/")
        parsed = urlparse(raw_url)
        if parsed.scheme not in ("http", "https"):
            raise ApiError(
                ApiErrorCode.INVALID_ARGUMENT,
                f"ASTRAWEAVE_OLLAMA_BASE_URL must use http or https scheme, got: {parsed.scheme!r}",
            )
        hostname = (parsed.hostname or "").lower()
        is_loopback = hostname in self._LOOPBACK_HOSTS or hostname.startswith("127.")
        if not is_loopback:
            self._logger.warning(
                "Ollama base URL %r does not resolve to a loopback address; "
                "ensure this is intentional to avoid SSRF risks.",
                raw_url,
            )
        self._base_url = raw_url
        self._timeout_seconds = timeout_seconds if timeout_seconds is not None else _get_ollama_timeout()
        self._transport = transport or _post_json

    def load_model(
        self,
        model_name: str,
        *,
        runtime_profile: str | None = None,
        backend_options: Mapping[str, Any] | None = None,
    ) -> InferenceModelBinding:
        tuning = resolve_runtime_tuning(
            model_name,
            runtime_profile=runtime_profile,
            backend_options=backend_options,
        )
        return InferenceModelBinding(
            backend=self.backend_name,
            requested_model_name=model_name,
            resolved_model_name=model_name,
            metadata={
                "base_url": self._base_url,
                "supports_prompt_generation": True,
                "transport": "http",
                "runtime_profile": tuning.profile_name,
                "model_size_billion": tuning.model_size_billion,
                "model_size_label": tuning.model_size_label,
                "backend_options": tuning.backend_options,
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
        backend_options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
        }
        if system_prompt:
            payload["system"] = system_prompt
        options = merge_backend_options(backend_options)
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if temperature is not None:
            options["temperature"] = temperature
        if options:
            payload["options"] = options

        raw = self._transport(f"{self._base_url}/api/generate", payload, self._timeout_seconds)
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
            "raw": {
                key: value
                for key, value in raw.items()
                if key not in {"prompt", "context"}
            },
        }


def create_inference_runtime(backend_name: str) -> InferenceRuntime:
    """Create a runtime adapter by stable backend name."""

    normalized = (backend_name or "").strip().lower()
    if normalized in {"", "simulation", "sim"}:
        return SimulationInferenceRuntime()
    if normalized in {"ollama", "ollama_local"}:
        return OllamaInferenceRuntime()
    raise ApiError(ApiErrorCode.INVALID_ARGUMENT, f"unsupported inference backend: {backend_name}")


def resolve_backend_and_model_name(
    model_name: str,
    runtime_backend: str | None = None,
) -> tuple[str, str]:
    """Resolve backend + model from explicit selector and/or model-name prefix.

    Resolution rules:
    - Explicit runtime backend wins when provided (except `auto`).
    - Otherwise model-name prefix (`ollama:` / `simulation:`) decides.
    - If nothing is specified, default backend is `simulation`.
    """

    if not model_name or not model_name.strip():
        raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "model_name must not be empty")

    normalized_model = model_name.strip()
    prefixed_backend: str | None = None
    resolved_model = normalized_model
    for prefix, backend in (("ollama:", "ollama"), ("simulation:", "simulation")):
        if normalized_model.lower().startswith(prefix):
            candidate = normalized_model[len(prefix):].strip()
            if not candidate:
                raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "model_name prefix must include a model identifier")
            prefixed_backend = backend
            resolved_model = candidate
            break

    backend_override = None
    if runtime_backend is not None:
        backend_override = runtime_backend.strip().lower()
        if backend_override in {"", "auto"}:
            backend_override = None
        elif backend_override not in {"simulation", "sim", "ollama", "ollama_local"}:
            raise ApiError(ApiErrorCode.INVALID_ARGUMENT, f"unsupported runtime_backend: {runtime_backend}")

    if backend_override is not None:
        if backend_override in {"sim", "simulation"}:
            return "simulation", resolved_model
        return "ollama", resolved_model
    if prefixed_backend is not None:
        return prefixed_backend, resolved_model
    return "simulation", resolved_model


def _post_json(url: str, payload: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    ssl_context = ssl.create_default_context() if url.startswith("https://") else None
    try:
        with request.urlopen(req, timeout=timeout_seconds, context=ssl_context) as response:
            body = response.read().decode("utf-8")
    except error.HTTPError as exc:  # pragma: no cover - network boundary
        detail = exc.read().decode("utf-8", errors="replace")
        raise ApiError(ApiErrorCode.INTERNAL, f"Ollama request failed with HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:  # pragma: no cover - network boundary
        raise ApiError(ApiErrorCode.INTERNAL, f"Ollama endpoint is unavailable: {exc.reason}") from exc
    except TimeoutError as exc:  # pragma: no cover - network boundary
        raise ApiError(ApiErrorCode.TIMEOUT, "Ollama request timed out") from exc

    try:
        decoded = json.loads(body)
    except json.JSONDecodeError as exc:  # pragma: no cover - network boundary
        raise ApiError(ApiErrorCode.INTERNAL, "Ollama returned invalid JSON") from exc
    if not isinstance(decoded, dict):
        raise ApiError(ApiErrorCode.INTERNAL, "Ollama response must be a JSON object")
    return decoded


__all__ = [
    "DEFAULT_OLLAMA_BASE_URL",
    "DEFAULT_OLLAMA_TIMEOUT_SECONDS",
    "InferenceModelBinding",
    "InferenceRuntime",
    "InferenceTransport",
    "OllamaInferenceRuntime",
    "SimulationInferenceRuntime",
    "create_inference_runtime",
    "resolve_backend_and_model_name",
]
