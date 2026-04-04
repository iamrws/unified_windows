"""Quantization provider framework for AstraWeave.

Provides pluggable quantization backends for KV cache compression.
Phase 7 ships simulated providers that model TurboQuant and FP8
compression ratios without requiring real CUDA kernels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import os
from typing import Any, Mapping, Protocol


class QuantizationBackend(str, Enum):
    """Supported quantization backends."""

    NONE = "none"
    FP8 = "fp8"
    TURBOQUANT = "turboquant"
    TQ1_0 = "tq1_0"
    TQ2_0 = "tq2_0"


@dataclass(frozen=True, slots=True)
class QuantizationResult:
    """Result of applying quantization to a tensor or KV block."""

    backend: QuantizationBackend
    original_bytes: int
    compressed_bytes: int
    compression_ratio: float
    bit_width: float
    metadata: dict[str, Any] = field(default_factory=dict)


class QuantizationProvider(Protocol):
    """Protocol for pluggable quantization backends."""

    @property
    def backend_name(self) -> QuantizationBackend: ...

    def supported_bit_widths(self) -> tuple[float, ...]: ...

    def estimate_compression_ratio(
        self,
        original_bytes: int,
        *,
        bit_width: float | None = None,
    ) -> float: ...

    def quantize(
        self,
        tensor_id: str,
        original_bytes: int,
        *,
        bit_width: float | None = None,
    ) -> QuantizationResult: ...


class SimulatedTurboQuantProvider:
    """Simulated TurboQuant provider for orchestration planning."""

    @property
    def backend_name(self) -> QuantizationBackend:
        return QuantizationBackend.TURBOQUANT

    def supported_bit_widths(self) -> tuple[float, ...]:
        return (2.5, 3.0, 3.5, 4.0)

    def estimate_compression_ratio(
        self,
        original_bytes: int,
        *,
        bit_width: float | None = None,
    ) -> float:
        effective_bits = bit_width if bit_width is not None else 3.5
        return 32.0 / effective_bits

    def quantize(
        self,
        tensor_id: str,
        original_bytes: int,
        *,
        bit_width: float | None = None,
    ) -> QuantizationResult:
        effective_bits = bit_width if bit_width is not None else 3.5
        ratio = self.estimate_compression_ratio(original_bytes, bit_width=effective_bits)
        compressed = max(1, int(original_bytes / ratio))
        return QuantizationResult(
            backend=self.backend_name,
            original_bytes=original_bytes,
            compressed_bytes=compressed,
            compression_ratio=ratio,
            bit_width=effective_bits,
            metadata={
                "algorithm": "turboquant_simulated",
                "stages": ["polarquant", "qjl_residual"],
            },
        )


class FP8Provider:
    """FP8 (E4M3/E5M2) quantization provider."""

    @property
    def backend_name(self) -> QuantizationBackend:
        return QuantizationBackend.FP8

    def supported_bit_widths(self) -> tuple[float, ...]:
        return (8.0,)

    def estimate_compression_ratio(
        self,
        original_bytes: int,
        *,
        bit_width: float | None = None,
    ) -> float:
        return 4.0

    def quantize(
        self,
        tensor_id: str,
        original_bytes: int,
        *,
        bit_width: float | None = None,
    ) -> QuantizationResult:
        ratio = self.estimate_compression_ratio(original_bytes, bit_width=bit_width)
        compressed = max(1, int(original_bytes / ratio))
        return QuantizationResult(
            backend=self.backend_name,
            original_bytes=original_bytes,
            compressed_bytes=compressed,
            compression_ratio=ratio,
            bit_width=8.0,
            metadata={"format": "e4m3"},
        )


class NoneProvider:
    """Pass-through provider that applies no quantization."""

    @property
    def backend_name(self) -> QuantizationBackend:
        return QuantizationBackend.NONE

    def supported_bit_widths(self) -> tuple[float, ...]:
        return (32.0,)

    def estimate_compression_ratio(
        self,
        original_bytes: int,
        *,
        bit_width: float | None = None,
    ) -> float:
        return 1.0

    def quantize(
        self,
        tensor_id: str,
        original_bytes: int,
        *,
        bit_width: float | None = None,
    ) -> QuantizationResult:
        return QuantizationResult(
            backend=self.backend_name,
            original_bytes=original_bytes,
            compressed_bytes=original_bytes,
            compression_ratio=1.0,
            bit_width=32.0,
        )


class TQ1_0Provider:
    """TQ1_0 (ternary base-3) quantization provider."""

    @property
    def backend_name(self) -> QuantizationBackend:
        return QuantizationBackend.TQ1_0

    def supported_bit_widths(self) -> tuple[float, ...]:
        return (1.6875,)

    def estimate_compression_ratio(
        self,
        original_bytes: int,
        *,
        bit_width: float | None = None,
    ) -> float:
        return 32.0 / 1.6875

    def quantize(
        self,
        tensor_id: str,
        original_bytes: int,
        *,
        bit_width: float | None = None,
    ) -> QuantizationResult:
        ratio = self.estimate_compression_ratio(original_bytes)
        compressed = max(1, int(original_bytes / ratio))
        return QuantizationResult(
            backend=self.backend_name,
            original_bytes=original_bytes,
            compressed_bytes=compressed,
            compression_ratio=ratio,
            bit_width=1.6875,
            metadata={
                "ggml_type_id": 34,
                "block_size": 256,
                "block_bytes": 42,
                "encoding": "ternary_base3",
            },
        )


class TQ2_0Provider:
    """TQ2_0 (2-bit polar) quantization provider."""

    @property
    def backend_name(self) -> QuantizationBackend:
        return QuantizationBackend.TQ2_0

    def supported_bit_widths(self) -> tuple[float, ...]:
        return (2.0625,)

    def estimate_compression_ratio(
        self,
        original_bytes: int,
        *,
        bit_width: float | None = None,
    ) -> float:
        return 32.0 / 2.0625

    def quantize(
        self,
        tensor_id: str,
        original_bytes: int,
        *,
        bit_width: float | None = None,
    ) -> QuantizationResult:
        ratio = self.estimate_compression_ratio(original_bytes)
        compressed = max(1, int(original_bytes / ratio))
        return QuantizationResult(
            backend=self.backend_name,
            original_bytes=original_bytes,
            compressed_bytes=compressed,
            compression_ratio=ratio,
            bit_width=2.0625,
            metadata={
                "ggml_type_id": 35,
                "block_size": 256,
                "block_bytes": 66,
                "encoding": "2bit_polar",
            },
        )


# ---------------------------------------------------------------------------
# Tier-aware provider selection
# ---------------------------------------------------------------------------

# Default mapping (8 GB VRAM class)
_TIER_PROVIDERS_8GB: dict[str, type] = {
    "HOT": SimulatedTurboQuantProvider,
    "WARM": FP8Provider,
    "COLD": NoneProvider,
}

# High-VRAM mapping (32 GB+ VRAM with real TQ CUDA kernels)
_TIER_PROVIDERS_32GB: dict[str, type] = {
    "HOT": TQ2_0Provider,
    "WARM": TQ1_0Provider,
    "COLD": FP8Provider,
}

_TIER_PROVIDER_PROFILES: dict[str, dict[str, type]] = {
    "default": _TIER_PROVIDERS_8GB,
    "high_vram": _TIER_PROVIDERS_32GB,
}

_PROVIDER_BY_BACKEND: dict[str, type] = {
    QuantizationBackend.NONE.value: NoneProvider,
    QuantizationBackend.FP8.value: FP8Provider,
    QuantizationBackend.TURBOQUANT.value: SimulatedTurboQuantProvider,
    QuantizationBackend.TQ1_0.value: TQ1_0Provider,
    QuantizationBackend.TQ2_0.value: TQ2_0Provider,
}

_TIER_PROVIDER_MAP_ENV = "ASTRAWEAVE_TIER_PROVIDER_MAP"
_VRAM_BUDGET_ENV = "ASTRAWEAVE_VRAM_BUDGET_BYTES"
_32GB_BYTES = 32 * 1024**3


def _effective_vram_budget_bytes(vram_budget_bytes: int | None) -> int | None:
    if vram_budget_bytes is not None:
        return vram_budget_bytes if vram_budget_bytes > 0 else None

    raw = os.environ.get(_VRAM_BUDGET_ENV)
    if raw is None:
        return None
    try:
        parsed = int(raw.strip())
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _parse_mapping_override(raw_mapping: str | None) -> dict[str, str]:
    if raw_mapping is None:
        return {}

    entries = [entry.strip() for entry in raw_mapping.split(",") if entry.strip()]
    parsed: dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            continue
        tier_text, backend_text = entry.split("=", 1)
        tier_key = tier_text.strip().upper()
        backend_value = backend_text.strip().lower()
        if tier_key not in {"HOT", "WARM", "COLD"}:
            continue
        if backend_value not in _PROVIDER_BY_BACKEND:
            continue
        parsed[tier_key] = backend_value
    return parsed


def default_tier_provider_mapping(
    vram_budget_bytes: int | None = None,
    *,
    profile: str = "default",
) -> dict[str, str]:
    """Return default tier->backend mapping for the active VRAM class."""

    if profile == "high_vram":
        mapping = _TIER_PROVIDERS_32GB
    else:
        effective_budget = _effective_vram_budget_bytes(vram_budget_bytes)
        mapping = _TIER_PROVIDERS_32GB if effective_budget is not None and effective_budget >= _32GB_BYTES else _TIER_PROVIDERS_8GB
    return {
        tier: cls().backend_name.value
        for tier, cls in mapping.items()
    }


def resolve_tier_provider_mapping(
    *,
    vram_budget_bytes: int | None = None,
    mapping_override: Mapping[str, str] | None = None,
    profile: str = "default",
) -> dict[str, str]:
    """Resolve the active tier->backend mapping."""

    resolved = default_tier_provider_mapping(vram_budget_bytes=vram_budget_bytes, profile=profile)
    resolved.update(_parse_mapping_override(os.environ.get(_TIER_PROVIDER_MAP_ENV)))

    if mapping_override:
        for tier, backend_name in mapping_override.items():
            if not isinstance(tier, str) or not isinstance(backend_name, str):
                continue
            tier_key = tier.strip().upper()
            backend_value = backend_name.strip().lower()
            if tier_key in {"HOT", "WARM", "COLD"} and backend_value in _PROVIDER_BY_BACKEND:
                resolved[tier_key] = backend_value

    return resolved


def provider_for_backend(backend_name: str) -> QuantizationProvider:
    """Return a provider instance for a stable backend name."""

    normalized = (backend_name or "").strip().lower()
    cls = _PROVIDER_BY_BACKEND.get(normalized, NoneProvider)
    return cls()


def provider_for_tier(
    tier: str,
    *,
    profile: str = "default",
    custom_mapping: Mapping[str, type] | None = None,
    vram_budget_bytes: int | None = None,
    mapping_override: Mapping[str, str] | None = None,
) -> QuantizationProvider:
    """Return the quantization provider for a memory tier."""

    tier_key = tier.upper()

    if custom_mapping is not None:
        cls = custom_mapping.get(tier_key, NoneProvider)
        return cls()

    if mapping_override is not None or os.environ.get(_TIER_PROVIDER_MAP_ENV) is not None:
        backend_name = resolve_tier_provider_mapping(
            vram_budget_bytes=vram_budget_bytes,
            mapping_override=mapping_override,
            profile=profile,
        ).get(tier_key, QuantizationBackend.NONE.value)
        return provider_for_backend(backend_name)

    if profile not in _TIER_PROVIDER_PROFILES:
        profile = "default"

    if profile == "default" and vram_budget_bytes is not None and vram_budget_bytes >= _32GB_BYTES:
        profile = "high_vram"

    mapping = _TIER_PROVIDER_PROFILES.get(profile, _TIER_PROVIDERS_8GB)
    cls = mapping.get(tier_key, NoneProvider)
    return cls()


__all__ = [
    "FP8Provider",
    "NoneProvider",
    "QuantizationBackend",
    "QuantizationProvider",
    "QuantizationResult",
    "SimulatedTurboQuantProvider",
    "TQ1_0Provider",
    "TQ2_0Provider",
    "default_tier_provider_mapping",
    "provider_for_backend",
    "provider_for_tier",
    "resolve_tier_provider_mapping",
]
