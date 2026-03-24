"""Runtime tuning helpers for local inference backends."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
import re
from typing import Any, Mapping

from .errors import ApiError, ApiErrorCode

DEFAULT_RUNTIME_PROFILE = "default"
AUTO_RUNTIME_PROFILE = "auto"
VRAM_CONSTRAINED_RUNTIME_PROFILE = "vram_constrained"

_PROFILE_ALIASES = {
    "": AUTO_RUNTIME_PROFILE,
    "adaptive": AUTO_RUNTIME_PROFILE,
    "auto": AUTO_RUNTIME_PROFILE,
    "balanced": DEFAULT_RUNTIME_PROFILE,
    "constrained": VRAM_CONSTRAINED_RUNTIME_PROFILE,
    "default": DEFAULT_RUNTIME_PROFILE,
    "memory_saver": VRAM_CONSTRAINED_RUNTIME_PROFILE,
    "low_vram": VRAM_CONSTRAINED_RUNTIME_PROFILE,
    "offload": VRAM_CONSTRAINED_RUNTIME_PROFILE,
    "ram_offload": VRAM_CONSTRAINED_RUNTIME_PROFILE,
    "standard": DEFAULT_RUNTIME_PROFILE,
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


def is_large_model(model_size_billion: float | None) -> bool:
    """Return whether a model is in the constrained-VRAM class."""

    return model_size_billion is not None and model_size_billion >= 14.0


def default_runtime_profile_for_model(model_size_billion: float | None) -> str:
    """Select the default runtime profile for a model scale."""

    return VRAM_CONSTRAINED_RUNTIME_PROFILE if is_large_model(model_size_billion) else DEFAULT_RUNTIME_PROFILE


def profile_backend_options(profile_name: str, model_size_billion: float | None) -> dict[str, Any]:
    """Return deterministic backend options for a runtime profile."""

    profile = normalize_runtime_profile_name(profile_name)
    if profile == AUTO_RUNTIME_PROFILE:
        profile = default_runtime_profile_for_model(model_size_billion)
    if profile == DEFAULT_RUNTIME_PROFILE:
        return {}
    if profile != VRAM_CONSTRAINED_RUNTIME_PROFILE:
        raise ApiError(ApiErrorCode.INVALID_ARGUMENT, f"unsupported runtime profile: {profile_name}")
    return _vram_constrained_backend_options(model_size_billion)


def resolve_runtime_tuning(
    model_name: str,
    *,
    runtime_profile: str | None = None,
    backend_options: Mapping[str, Any] | None = None,
) -> RuntimeTuning:
    """Resolve a stable runtime profile and effective backend option payload."""

    model_size_billion = infer_model_size_billion(model_name)
    requested_profile = normalize_runtime_profile_name(runtime_profile)
    resolved_profile = (
        default_runtime_profile_for_model(model_size_billion)
        if requested_profile == AUTO_RUNTIME_PROFILE
        else requested_profile
    )
    profile_options = profile_backend_options(resolved_profile, model_size_billion)
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


def _vram_constrained_backend_options(model_size_billion: float | None) -> dict[str, Any]:
    if model_size_billion is None or model_size_billion < 14.0:
        num_ctx = 2048
        num_batch = 32
    elif model_size_billion < 30.0:
        num_ctx = 2048
        num_batch = 24
    elif model_size_billion < 70.0:
        num_ctx = 1536
        num_batch = 16
    else:
        num_ctx = 1024
        num_batch = 8

    num_keep = max(16, num_ctx // 64)
    return {
        "f16_kv": True,
        "low_vram": True,
        "num_batch": num_batch,
        "num_ctx": num_ctx,
        "num_keep": num_keep,
    }


__all__ = [
    "AUTO_RUNTIME_PROFILE",
    "DEFAULT_RUNTIME_PROFILE",
    "RuntimeTuning",
    "VRAM_CONSTRAINED_RUNTIME_PROFILE",
    "default_runtime_profile_for_model",
    "format_model_size_label",
    "infer_model_size_billion",
    "is_large_model",
    "merge_backend_options",
    "normalize_backend_options",
    "normalize_runtime_profile_name",
    "profile_backend_options",
    "resolve_runtime_tuning",
]
