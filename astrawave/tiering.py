"""Tiering and placement planning for AstraWeave."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .types import MemoryTier


@dataclass(frozen=True, slots=True)
class PlacementRequest:
    """Description of a resource that needs residency placement."""

    resource_id: str
    bytes_required: int
    is_active: bool = False
    reuse_score: float = 0.0
    preferred_tier: MemoryTier | None = None
    requires_hot_kernel: bool = False


@dataclass(frozen=True, slots=True)
class PlacementDecision:
    """Placement outcome for a single resource."""

    resource_id: str
    tier: MemoryTier
    bytes_required: int
    reason_code: str


@dataclass(frozen=True, slots=True)
class PlacementPlan:
    """Complete placement plan returned by the planner."""

    decisions: tuple[PlacementDecision, ...]
    hot_bytes: int
    warm_bytes: int
    cold_bytes: int

    def by_tier(self, tier: MemoryTier) -> tuple[PlacementDecision, ...]:
        """Return all decisions assigned to the requested tier."""

        return tuple(decision for decision in self.decisions if decision.tier == tier)


@dataclass(frozen=True, slots=True)
class PlacementPolicy:
    """Thresholds used by the placement planner."""

    hot_reuse_score: float = 0.75
    warm_reuse_score: float = 0.35
    hot_headroom_ratio: float = 0.20


def dynamic_headroom_ratio(vram_budget_gb: float | None = None) -> float:
    """Compute headroom ratio as a function of VRAM budget."""

    if vram_budget_gb is None or vram_budget_gb <= 0:
        return 0.20
    return max(0.05, min(0.20, 4.0 / vram_budget_gb))


def policy_for_vram_budget(vram_budget_gb: float | None = None) -> PlacementPolicy:
    """Create a PlacementPolicy tuned for the supplied VRAM budget."""

    return PlacementPolicy(hot_headroom_ratio=dynamic_headroom_ratio(vram_budget_gb))


def hot_headroom_ratio_for_budget(vram_budget_bytes: int) -> float:
    """Byte-oriented compatibility helper for headroom ratio calculation."""

    return dynamic_headroom_ratio(vram_budget_bytes / float(1024**3))


class PlacementPlanner:
    """Deterministic planner that classifies resources into HOT/WARM/COLD."""

    def __init__(
        self,
        policy: PlacementPolicy | None = None,
        *,
        hot_kernel_available: bool = True,
    ) -> None:
        self._policy = policy or PlacementPolicy()
        self._hot_kernel_available = hot_kernel_available

    @property
    def policy(self) -> PlacementPolicy:
        """Expose the planner policy for diagnostics."""

        return self._policy

    def classify(
        self,
        request: PlacementRequest,
        hot_budget_bytes: int,
        hot_used_bytes: int = 0,
        *,
        hot_compression_ratio: float = 1.0,
        cuda_kernels_available: bool | None = None,
        hot_kernel_available: bool | None = None,
    ) -> PlacementDecision:
        """Classify one resource into a residency tier."""

        kernels_available = self._hot_kernel_available
        if cuda_kernels_available is not None:
            kernels_available = cuda_kernels_available
        if hot_kernel_available is not None:
            kernels_available = hot_kernel_available

        if (not kernels_available and hot_compression_ratio > 1.0) or (
            request.requires_hot_kernel and not kernels_available
        ):
            return PlacementDecision(
                resource_id=request.resource_id,
                tier=MemoryTier.WARM,
                bytes_required=request.bytes_required,
                reason_code="PLACEMENT_WARM_NO_CUDA_KERNELS",
            )

        effective_bytes = (
            max(1, int(request.bytes_required / hot_compression_ratio))
            if hot_compression_ratio > 1.0
            else request.bytes_required
        )

        if request.preferred_tier is not None:
            tier = request.preferred_tier
            reason_code = "PLACEMENT_PREFERRED_TIER"
        elif request.is_active or request.reuse_score >= self._policy.hot_reuse_score:
            if hot_used_bytes + effective_bytes <= self._reserve_limit(hot_budget_bytes):
                tier = MemoryTier.HOT
                reason_code = "PLACEMENT_HOT_ACTIVE_OR_HIGH_REUSE"
            else:
                tier = MemoryTier.WARM
                reason_code = "PLACEMENT_WARM_HOT_HEADROOM_EXCEEDED"
        elif request.reuse_score >= self._policy.warm_reuse_score:
            tier = MemoryTier.WARM
            reason_code = "PLACEMENT_WARM_REUSE_SCORE"
        else:
            tier = MemoryTier.COLD
            reason_code = "PLACEMENT_COLD_LOW_REUSE"

        return PlacementDecision(
            resource_id=request.resource_id,
            tier=tier,
            bytes_required=request.bytes_required,
            reason_code=reason_code,
        )

    def plan(
        self,
        requests: Sequence[PlacementRequest],
        hot_budget_bytes: int,
        *,
        hot_compression_ratio: float = 1.0,
        cuda_kernels_available: bool | None = None,
        hot_kernel_available: bool | None = None,
    ) -> PlacementPlan:
        """Create a deterministic placement plan for a collection of resources."""

        decisions: list[PlacementDecision] = []
        hot_used_bytes = 0
        warm_bytes = 0
        cold_bytes = 0

        for request in requests:
            decision = self.classify(
                request,
                hot_budget_bytes=hot_budget_bytes,
                hot_used_bytes=hot_used_bytes,
                hot_compression_ratio=hot_compression_ratio,
                cuda_kernels_available=cuda_kernels_available,
                hot_kernel_available=hot_kernel_available,
            )
            decisions.append(decision)

            effective = (
                max(1, int(decision.bytes_required / hot_compression_ratio))
                if decision.tier == MemoryTier.HOT and hot_compression_ratio > 1.0
                else decision.bytes_required
            )

            if decision.tier == MemoryTier.HOT:
                hot_used_bytes += effective
            elif decision.tier == MemoryTier.WARM:
                warm_bytes += decision.bytes_required
            else:
                cold_bytes += decision.bytes_required

        return PlacementPlan(
            decisions=tuple(decisions),
            hot_bytes=hot_used_bytes,
            warm_bytes=warm_bytes,
            cold_bytes=cold_bytes,
        )

    def summarize(self, plan: PlacementPlan) -> Mapping[MemoryTier, int]:
        """Summarize a plan into per-tier byte totals."""

        return {
            MemoryTier.HOT: plan.hot_bytes,
            MemoryTier.WARM: plan.warm_bytes,
            MemoryTier.COLD: plan.cold_bytes,
        }

    def _reserve_limit(self, hot_budget_bytes: int) -> int:
        """Reserve a fraction of HOT budget to avoid decode-time spikes."""

        reserve = int(hot_budget_bytes * self._policy.hot_headroom_ratio)
        return max(hot_budget_bytes - reserve, 0)


__all__ = [
    "PlacementDecision",
    "PlacementPlan",
    "PlacementPlanner",
    "PlacementPolicy",
    "PlacementRequest",
    "dynamic_headroom_ratio",
    "hot_headroom_ratio_for_budget",
    "policy_for_vram_budget",
]
