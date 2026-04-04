"""Runtime tuning helpers for local inference backends."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
import os
import re
from typing import Any, Mapping

from .errors import ApiError, ApiErrorCode

DEFAULT_RUNTIME_PROFILE = "default"
AUTO_RUNTIME_PROFILE = "auto"
VRAM_CONSTRAINED_RUNTIME_PROFILE = "vram_constrained"
THROUGHPUT_RUNTIME_PROFILE = "throughput"

DEFAULT_VRAM_BUDGET_BYTES = 8 * 1024**3
_VRAM_BUDGET_ENV = "ASTRAWEAVE_VRAM_BUDGET_BYTES"

# Valid KV cache quantization types (maps to llama.cpp type_k / type_v)
VALID_KV_TYPES = frozenset({"f16", "f32", "q8_0", "q4_0", "tq1_0", "tq2_0"})
_QUANTIZED_KV_TYPES = VALID_KV_TYPES - {"f16", "f32"}

_PROFILE_ALIASES = {
    "": AUTO_RUNTIME_PROFILE,
    "adaptive": AUTO_RUNTIME_PROFILE,
    "auto": AUTO_RUNTIME_PROFILE,
    "balanced": DEFAULT_RUNTIME_PROFILE,
    "constrained": VRAM_CONSTRAINED_RUNTIME_PROFILE,
    "default": DEFAULT_RUNTIME_PROFILE,
    "fast": THROUGHPUT_RUNTIME_PROFILE,
    "high_throughput": THROUGHPUT_RUNTIME_PROFILE,
    "memory_saver": VRAM_CONSTRAINED_RUNTIME_PROFILE,
    "low_vram": VRAM_CONSTRAINED_RUNTIME_PROFILE,
    "offload": VRAM_CONSTRAINED_RUNTIME_PROFILE,
    "ram_offload": VRAM_CONSTRAINED_RUNTIME_PROFILE,
    "standard": DEFAULT_RUNTIME_PROFILE,
    "throughput": THROUGHPUT_RUNTIME_PROFILE,
    "vram_constrained": VRAM_CONSTRAINED_RUNTIME_PROFILE,
}


@dataclass(frozen=True, slots=True)
class RuntimeTuning:
    """Resolved tuning metadata for a model binding."""

    profile_name: str
    model_size_billion: float | None
    model_size_label: str | None
    backend_options: dict[str, Any]


def infer_model_size_billion(model_name: str) -> float | None:
    """Infer an approximate model scale from a model tag."""

    if not isinstance(model_name, str):
        return None

    normalized = model_name.strip().lower()
    if not normalized:
        return None

    candidates: list[float] = []

    for match in re.finditer(r"(?<!\d)(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)\s*[bB](?!\w)", normalized):
        lhs = float(match.group(1))
        rhs = float(match.group(2))
        candidates.append(lhs * rhs)

    for match in re.finditer(r"(?<!\d)(\d+(?:\.\d+)?)\s*[bB](?!\w)", normalized):
        candidates.append(float(match.group(1)))

    return max(candidates) if candidates else None


def format_model_size_label(model_size_billion: float | None) -> str | None:
    """Render a compact human-readable size label."""

    if model_size_billion is None:
        return None
    if float(model_size_billion).is_integer():
        return f"{int(model_size_billion)}b"
    return f"{model_size_billion:g}b"


def normalize_runtime_profile_name(profile_name: str | None) -> str:
    """Normalize a user-supplied runtime profile name."""

    normalized = "auto" if profile_name is None else profile_name.strip().lower().replace("-", "_")
    if normalized in _PROFILE_ALIASES:
        return _PROFILE_ALIASES[normalized]
    raise ApiError(ApiErrorCode.INVALID_ARGUMENT, f"unsupported runtime profile: {profile_name}")


def resolve_vram_budget_bytes(vram_budget_bytes: int | None = None) -> int:
    """Resolve effective VRAM budget bytes from override or environment."""

    if vram_budget_bytes is not None:
        if not isinstance(vram_budget_bytes, int) or isinstance(vram_budget_bytes, bool) or vram_budget_bytes <= 0:
            raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "vram_budget_bytes must be a positive integer when provided")
        return vram_budget_bytes

    raw = os.environ.get(_VRAM_BUDGET_ENV)
    if raw is None:
        return DEFAULT_VRAM_BUDGET_BYTES
    try:
        parsed = int(raw.strip())
    except ValueError:
        return DEFAULT_VRAM_BUDGET_BYTES
    if parsed <= 0:
        return DEFAULT_VRAM_BUDGET_BYTES
    return parsed


def large_model_threshold(vram_budget_gb: float | None = None) -> float:
    """Return the model-size threshold (billions) above which VRAM is constrained.

    Scales with available VRAM: 14B at 8 GB, 34B at 32 GB, 70B at 80 GB.
    Falls back to 14B when *vram_budget_gb* is None (legacy 8 GB default).
    """

    if vram_budget_gb is None or vram_budget_gb <= 8.0:
        return 14.0
    return min(70.0, 14.0 + (vram_budget_gb - 8.0) * (56.0 / 72.0))


def large_model_threshold_billion(vram_budget_bytes: int | None = None) -> float:
    """Byte-oriented companion to ``large_model_threshold`` for compatibility."""

    budget_bytes = resolve_vram_budget_bytes(vram_budget_bytes)
    if budget_bytes >= 32 * 1024**3:
        return 34.0
    if budget_bytes >= 16 * 1024**3:
        return 24.0
    return 14.0


def is_large_model(
    model_size_billion: float | None,
    vram_budget_gb: float | None = None,
    *,
    vram_budget_bytes: int | None = None,
) -> bool:
    """Return whether a model is in the constrained-VRAM class."""

    if model_size_billion is None:
        return False
    if vram_budget_bytes is not None:
        return model_size_billion >= large_model_threshold_billion(vram_budget_bytes)
    return model_size_billion >= large_model_threshold(vram_budget_gb)


def default_runtime_profile_for_model(
    model_size_billion: float | None,
    *,
    vram_budget_bytes: int | None = None,
) -> str:
    """Select the default runtime profile for a model scale."""

    return (
        VRAM_CONSTRAINED_RUNTIME_PROFILE
        if is_large_model(model_size_billion, vram_budget_bytes=vram_budget_bytes)
        else DEFAULT_RUNTIME_PROFILE
    )


def _normalize_flash_attn_value(value: Any) -> bool | str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "auto":
            return "auto"
        if normalized in {"true", "false"}:
            return normalized == "true"
    raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "backend option 'flash_attn' must be true, false, or 'auto'")


def validate_kv_cache_options(options: Mapping[str, Any]) -> None:
    """Validate KV cache type and flash attention option constraints."""

    type_k = options.get("type_k")
    type_v = options.get("type_v")
    flash_attn = _normalize_flash_attn_value(options.get("flash_attn"))

    if type_k is not None and type_k not in VALID_KV_TYPES:
        raise ApiError(ApiErrorCode.INVALID_ARGUMENT, f"unsupported type_k: {type_k}")
    if type_v is not None and type_v not in VALID_KV_TYPES:
        raise ApiError(ApiErrorCode.INVALID_ARGUMENT, f"unsupported type_v: {type_v}")
    if type_v in _QUANTIZED_KV_TYPES and flash_attn is False:
        raise ApiError(
            ApiErrorCode.INVALID_ARGUMENT,
            f"type_v={type_v} requires flash_attn to be enabled",
        )


def profile_backend_options(
    profile_name: str,
    model_size_billion: float | None,
    *,
    vram_budget_bytes: int | None = None,
) -> dict[str, Any]:
    """Return deterministic backend options for a runtime profile."""

    profile = normalize_runtime_profile_name(profile_name)
    if profile == AUTO_RUNTIME_PROFILE:
        profile = default_runtime_profile_for_model(model_size_billion, vram_budget_bytes=vram_budget_bytes)
    if profile == DEFAULT_RUNTIME_PROFILE:
        return {}
    if profile == THROUGHPUT_RUNTIME_PROFILE:
        return _throughput_backend_options(model_size_billion)
    if profile == VRAM_CONSTRAINED_RUNTIME_PROFILE:
        return _vram_constrained_backend_options(model_size_billion)
    raise ApiError(ApiErrorCode.INVALID_ARGUMENT, f"unsupported runtime profile: {profile_name}")


def resolve_runtime_tuning(
    model_name: str,
    *,
    runtime_profile: str | None = None,
    backend_options: Mapping[str, Any] | None = None,
    vram_budget_bytes: int | None = None,
) -> RuntimeTuning:
    """Resolve a stable runtime profile and effective backend option payload."""

    model_size_billion = infer_model_size_billion(model_name)
    requested_profile = normalize_runtime_profile_name(runtime_profile)
    resolved_profile = (
        default_runtime_profile_for_model(model_size_billion, vram_budget_bytes=vram_budget_bytes)
        if requested_profile == AUTO_RUNTIME_PROFILE
        else requested_profile
    )
    profile_options = profile_backend_options(
        resolved_profile,
        model_size_billion,
        vram_budget_bytes=vram_budget_bytes,
    )
    merged_backend_options = merge_backend_options(profile_options, backend_options)
    return RuntimeTuning(
        profile_name=resolved_profile,
        model_size_billion=model_size_billion,
        model_size_label=format_model_size_label(model_size_billion),
        backend_options=merged_backend_options,
    )


def normalize_backend_options(options: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate and normalize backend options to JSON-friendly values."""

    if options is None:
        return {}
    if not isinstance(options, Mapping):
        raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "backend_options must be a mapping or null")

    normalized: dict[str, Any] = {}
    for raw_key, value in options.items():
        if not isinstance(raw_key, str) or not raw_key.strip():
            raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "backend_options keys must be non-empty strings")
        key = raw_key.strip()
        normalized[key] = _normalize_backend_option_value(value, key)

    validate_kv_cache_options(normalized)
    return normalized


def merge_backend_options(*options: Mapping[str, Any] | None) -> dict[str, Any]:
    """Merge backend option mappings with later entries taking precedence."""

    merged: dict[str, Any] = {}
    for option_map in options:
        if option_map is None:
            continue
        merged.update(normalize_backend_options(option_map))
    return merged


def _normalize_backend_option_value(value: Any, key_path: str) -> Any:
    leaf_key = key_path.split(".")[-1]

    if leaf_key in {"type_k", "type_v"}:
        if not isinstance(value, str) or not value.strip():
            raise ApiError(ApiErrorCode.INVALID_ARGUMENT, f"backend option '{key_path}' must be a non-empty string")
        normalized_kv_type = value.strip().lower()
        if normalized_kv_type not in VALID_KV_TYPES:
            raise ApiError(
                ApiErrorCode.INVALID_ARGUMENT,
                f"backend option '{key_path}' must be one of {sorted(VALID_KV_TYPES)}",
            )
        return normalized_kv_type

    if leaf_key == "flash_attn":
        return _normalize_flash_attn_value(value)

    if leaf_key == "offload_kqv":
        if not isinstance(value, bool):
            raise ApiError(ApiErrorCode.INVALID_ARGUMENT, f"backend option '{key_path}' must be a boolean")
        return value

    if value is None:
        raise ApiError(ApiErrorCode.INVALID_ARGUMENT, f"backend option '{key_path}' cannot be null")
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ApiError(ApiErrorCode.INVALID_ARGUMENT, f"backend option '{key_path}' must be finite")
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for raw_key, nested_value in value.items():
            if not isinstance(raw_key, str) or not raw_key.strip():
                raise ApiError(ApiErrorCode.INVALID_ARGUMENT, f"backend option '{key_path}' has invalid nested key")
            nested_key = raw_key.strip()
            normalized[nested_key] = _normalize_backend_option_value(nested_value, f"{key_path}.{nested_key}")
        return normalized
    if isinstance(value, tuple):
        return [_normalize_backend_option_value(item, f"{key_path}[]") for item in value]
    if isinstance(value, list):
        return [_normalize_backend_option_value(item, f"{key_path}[]") for item in value]
    raise ApiError(
        ApiErrorCode.INVALID_ARGUMENT,
        f"backend option '{key_path}' has unsupported type: {type(value).__name__}",
    )


def _throughput_backend_options(model_size_billion: float | None) -> dict[str, Any]:
    """Backend options optimized for maximum tok/s on 32 GB+ VRAM systems."""

    if model_size_billion is None or model_size_billion < 14.0:
        num_ctx = 32768
        num_batch = 512
        type_k = "tq2_0"
        type_v = "f16"
    elif model_size_billion < 34.0:
        num_ctx = 8192
        num_batch = 256
        type_k = "tq2_0"
        type_v = "f16"
    elif model_size_billion < 70.0:
        num_ctx = 4096
        num_batch = 128
        type_k = "tq2_0"
        type_v = "f16"
    else:
        num_ctx = 2048
        num_batch = 64
        type_k = "tq1_0"
        type_v = "f16"

    num_keep = max(16, num_ctx // 64)
    return {
        "flash_attn": "auto",
        "num_batch": num_batch,
        "num_ctx": num_ctx,
        "num_gpu": -1,
        "num_keep": num_keep,
        "num_thread": 16,
        "offload_kqv": True,
        "type_k": type_k,
        "type_v": type_v,
    }


def _vram_constrained_backend_options(model_size_billion: float | None) -> dict[str, Any]:
    if model_size_billion is None or model_size_billion < 70.0:
        num_ctx = 2048
        num_batch = 32
    else:
        num_ctx = 1024
        num_batch = 16

    num_keep = max(16, num_ctx // 64)
    return {
        "flash_attn": False,
        "f16_kv": True,
        "low_vram": True,
        "num_batch": num_batch,
        "num_ctx": num_ctx,
        "num_keep": num_keep,
        "offload_kqv": False,
        "type_k": "f16",
        "type_v": "f16",
    }


__all__ = [
    "AUTO_RUNTIME_PROFILE",
    "DEFAULT_RUNTIME_PROFILE",
    "RuntimeTuning",
    "THROUGHPUT_RUNTIME_PROFILE",
    "VALID_KV_TYPES",
    "VRAM_CONSTRAINED_RUNTIME_PROFILE",
    "default_runtime_profile_for_model",
    "format_model_size_label",
    "infer_model_size_billion",
    "is_large_model",
    "large_model_threshold",
    "large_model_threshold_billion",
    "merge_backend_options",
    "normalize_backend_options",
    "normalize_runtime_profile_name",
    "profile_backend_options",
    "resolve_vram_budget_bytes",
    "resolve_runtime_tuning",
    "validate_kv_cache_options",
]
