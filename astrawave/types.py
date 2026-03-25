"""Core AstraWeave domain types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class MemoryTier(str, Enum):
    """Logical residency tiers used by the policy engine."""

    HOT = "HOT"
    WARM = "WARM"
    COLD = "COLD"


class ResidencyState(str, Enum):
    """Concrete residency state for a tensor or session-owned resource."""

    VRAM = "VRAM"
    PINNED_RAM = "PINNED_RAM"
    PAGEABLE_RAM = "PAGEABLE_RAM"
    CPU_ONLY = "CPU_ONLY"


class PolicyProfile(str, Enum):
    """Supported policy profiles."""

    STABILITY = "stability"
    THROUGHPUT = "throughput"


class SessionState(str, Enum):
    """Lifecycle states for a session."""

    NEW = "NEW"
    SESSION_CREATED = "SESSION_CREATED"
    MODEL_LOADED = "MODEL_LOADED"
    READY = "READY"
    RUNNING = "RUNNING"
    DEGRADED = "DEGRADED"
    CLOSED = "CLOSED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class PressureSnapshot:
    """Snapshot of the current memory pressure for a session."""

    session_id: str
    vram_budget_bytes: int
    vram_used_bytes: int
    pinned_ram_used_bytes: int
    pressure_level: float
    policy_profile: PolicyProfile
    timestamp_ms: int


@dataclass(frozen=True, slots=True)
class ResidencySnapshot:
    """Summarized residency view for a session."""

    session_id: str
    session_state: SessionState
    primary_tier: MemoryTier
    tensor_residency: Mapping[str, ResidencyState] = field(default_factory=dict)
    vram_bytes: int = 0
    pinned_ram_bytes: int = 0
    pageable_ram_bytes: int = 0
    cpu_only_bytes: int = 0
    active_run_in_progress: bool = False
    quantization_backend: str = "none"
    compression_ratio: float = 1.0


@dataclass(frozen=True, slots=True)
class TransferEvent:
    """Telemetry-friendly migration event."""

    session_id: str
    source: ResidencyState
    destination: ResidencyState
    bytes_moved: int
    reason_code: str
    timestamp_ms: int


@dataclass(frozen=True, slots=True)
class FallbackEvent:
    """Telemetry-friendly fallback event."""

    session_id: str
    step: str
    reason_code: str
    timestamp_ms: int

