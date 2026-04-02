"""Quantization provider framework for AstraWeave.

Provides pluggable quantization backends for KV cache compression.
Phase 7 ships simulated providers that model TurboQuant and FP8
compression ratios without requiring real CUDA kernels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


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


class TQ1_0Provider:
    """TQ1_0 (ternary base-3) quantization provider.

    Real GGML block structure: 256 elements in 42 bytes = 1.6875 bits/weight.
    Compression ratio from FP32: 32.0 / 1.6875 = 18.96x.
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
    """TQ2_0 (2-bit polar) quantization provider.

    Real GGML block structure: 256 elements in 66 bytes = 2.0625 bits/weight.
    Compression ratio from FP32: 32.0 / 2.0625 = 15.52x.
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


def provider_for_tier(
    tier: str,
    *,
    profile: str = "default",
    custom_mapping: dict[str, type] | None = None,
) -> QuantizationProvider:
    """Return the quantization provider for a memory tier.

    Default profile (8 GB VRAM):
        HOT  -> SimulatedTurboQuant (generic compression)
        WARM -> FP8 (moderate compression, RAM is cheaper)
        COLD -> None (pageable RAM, not worth the CPU cost)

    high_vram profile (32 GB+ VRAM with TQ CUDA kernels):
        HOT  -> TQ2_0 (15.5x, simpler kernel, low decode overhead)
        WARM -> TQ1_0 (19x, maximum compression for overflow)
        COLD -> FP8 (data may be promoted later)

    A *custom_mapping* dict overrides both profiles.
    """

    if custom_mapping is not None:
        mapping = custom_mapping
    else:
        mapping = _TIER_PROVIDER_PROFILES.get(profile, _TIER_PROVIDERS_8GB)
    cls = mapping.get(tier.upper(), NoneProvider)
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
    "provider_for_tier",
]
