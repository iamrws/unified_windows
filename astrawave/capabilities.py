"""Capability-mode detection for AstraWeave.

The service uses this module to choose a deterministic runtime path before any
policy decisions are made.  The detection API is intentionally pure and
dependency-injectable so callers can feed it either real hardware probes or
synthetic fixtures in tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class CapabilityMode(str, Enum):
    """Supported runtime capability modes."""

    NUMA_DGPU = "NUMA_dGPU"
    CACHE_COHERENT_UMA = "CacheCoherentUMA"
    UMA = "UMA"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True, slots=True)
class CapabilitySignals:
    """Hardware signals consumed by capability detection."""

    supports_runtime: bool
    has_discrete_gpu: bool
    is_uma: bool
    is_cache_coherent_uma: bool


@dataclass(frozen=True, slots=True)
class CapabilitySnapshot:
    """Resolved capability mode and the signals that produced it."""

    mode: CapabilityMode
    signals: CapabilitySignals
    reason_code: str


class CapabilityProbe(Protocol):
    """Protocol for pluggable capability probing."""

    def probe(self) -> CapabilitySignals:
        """Return the current hardware capability signals."""


def resolve_capability_mode(signals: CapabilitySignals) -> CapabilitySnapshot:
    """Map raw hardware signals to a deterministic runtime capability mode."""

    if not signals.supports_runtime:
        return CapabilitySnapshot(
            mode=CapabilityMode.UNSUPPORTED,
            signals=signals,
            reason_code="CAPABILITY_UNSUPPORTED_RUNTIME",
        )

    if signals.is_cache_coherent_uma:
        return CapabilitySnapshot(
            mode=CapabilityMode.CACHE_COHERENT_UMA,
            signals=signals,
            reason_code="CAPABILITY_CACHE_COHERENT_UMA",
        )

    if signals.is_uma:
        return CapabilitySnapshot(
            mode=CapabilityMode.UMA,
            signals=signals,
            reason_code="CAPABILITY_UMA",
        )

    if signals.has_discrete_gpu:
        return CapabilitySnapshot(
            mode=CapabilityMode.NUMA_DGPU,
            signals=signals,
            reason_code="CAPABILITY_NUMA_DGPU",
        )

    return CapabilitySnapshot(
        mode=CapabilityMode.UNSUPPORTED,
        signals=signals,
        reason_code="CAPABILITY_NO_SUPPORTED_GPU",
    )


def probe_capability_mode(probe: CapabilityProbe) -> CapabilitySnapshot:
    """Convenience helper that resolves a probe into a capability snapshot."""

    return resolve_capability_mode(probe.probe())
