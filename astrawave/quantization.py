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
    """Simulated TurboQuant provider for orchestration planning.

    Models the theoretical compression ratios from the TurboQuant paper
    (ICLR 2026) without requiring real CUDA kernels. Uses the ratio
    32 / bit_width where bit_width defaults to 3.5 (matching full-cache
    quality at 4.6x compression).
    """

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
    """FP8 (E4M3/E5M2) quantization provider.

    FP8 compresses from FP32 (4 bytes) to 1 byte per element: 4x compression.
    From FP16 (2 bytes) to 1 byte: 2x compression.
    This provider assumes FP32 baseline (4x).
    """

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
        return 4.0  # 32-bit -> 8-bit

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


class TQ2_0Provider:
    """TQ2_0 (2-bit polar) quantization provider.

    Based on the GGML block_tq2_0 structure: 66 bytes per 256 elements,
    yielding 2.0625 bits per weight and 15.52x compression from FP32.
    """

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
        return 32.0 / 2.0625  # 15.515...

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


class TQ1_0Provider:
    """TQ1_0 (ternary base-3) quantization provider.

    Based on the GGML block_tq1_0 structure: 42 bytes per 256 elements,
    yielding 1.6875 bits per weight and 18.96x compression from FP32.
    """

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
        return 32.0 / 1.6875  # 18.962...

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


# ---------------------------------------------------------------------------
# Tier-aware provider selection
# ---------------------------------------------------------------------------

_PROVIDER_BY_BACKEND: dict[str, type] = {
    QuantizationBackend.NONE.value: NoneProvider,
    QuantizationBackend.FP8.value: FP8Provider,
    QuantizationBackend.TURBOQUANT.value: SimulatedTurboQuantProvider,
    QuantizationBackend.TQ1_0.value: TQ1_0Provider,
    QuantizationBackend.TQ2_0.value: TQ2_0Provider,
}

_DEFAULT_TIER_MAPPING: dict[str, str] = {
    "HOT": QuantizationBackend.TQ2_0.value,
    "WARM": QuantizationBackend.TQ1_0.value,
    "COLD": QuantizationBackend.NONE.value,
}

_32GB_TIER_MAPPING: dict[str, str] = {
    "HOT": QuantizationBackend.TQ2_0.value,
    "WARM": QuantizationBackend.TQ1_0.value,
    "COLD": QuantizationBackend.FP8.value,
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


def default_tier_provider_mapping(vram_budget_bytes: int | None = None) -> dict[str, str]:
    """Return default tier->backend mapping for the active VRAM class."""

    effective_budget = _effective_vram_budget_bytes(vram_budget_bytes)
    if effective_budget is not None and effective_budget >= _32GB_BYTES:
        return dict(_32GB_TIER_MAPPING)
    return dict(_DEFAULT_TIER_MAPPING)


def resolve_tier_provider_mapping(
    *,
    vram_budget_bytes: int | None = None,
    mapping_override: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Resolve the active tier->backend mapping.

    Priority order (lowest to highest):
    1. budget-aware defaults
    2. env override (`ASTRAWEAVE_TIER_PROVIDER_MAP`)
    3. explicit `mapping_override`
    """

    resolved = default_tier_provider_mapping(vram_budget_bytes=vram_budget_bytes)
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
    vram_budget_bytes: int | None = None,
    mapping_override: Mapping[str, str] | None = None,
) -> QuantizationProvider:
    """Return the default quantization provider for a memory tier."""

    mapping = resolve_tier_provider_mapping(
        vram_budget_bytes=vram_budget_bytes,
        mapping_override=mapping_override,
    )
    backend_name = mapping.get(tier.upper(), QuantizationBackend.NONE.value)
    return provider_for_backend(backend_name)


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
