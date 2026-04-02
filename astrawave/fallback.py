"""Fallback ladder and anti-oscillation controls for AstraWeave."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence


class FallbackStep(str, Enum):
    """Mandatory fallback ladder steps in execution order."""

    KV_QUANTIZATION_UPGRADE = "kv_quantization_upgrade"
    KV_CONTEXT_REDUCTION = "kv_context_reduction"
    BATCH_REDUCTION = "batch_reduction"
    PRECISION_REDUCTION = "precision_reduction"
    SELECTIVE_CPU_OFFLOAD = "selective_cpu_offload"
    CONTROLLED_FAIL = "controlled_fail"


DEFAULT_FALLBACK_LADDER: tuple[FallbackStep, ...] = (
    FallbackStep.KV_QUANTIZATION_UPGRADE,
    FallbackStep.KV_CONTEXT_REDUCTION,
    FallbackStep.BATCH_REDUCTION,
    FallbackStep.PRECISION_REDUCTION,
    FallbackStep.SELECTIVE_CPU_OFFLOAD,
    FallbackStep.CONTROLLED_FAIL,
)


# ---------------------------------------------------------------------------
# KV Quantization Upgrade Progression (F-032)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class KVQuantizationLevel:
    """One step in the KV quantization upgrade progression."""

    type_k: str
    type_v: str
    label: str
    estimated_compression: float
    metadata: dict[str, Any] = field(default_factory=dict)


# Concrete progression: F16 -> TQ2_0 -> TQ1_0
KV_QUANTIZATION_PROGRESSION: tuple[KVQuantizationLevel, ...] = (
    KVQuantizationLevel(
        type_k="f16", type_v="f16", label="baseline",
        estimated_compression=1.0,
    ),
    KVQuantizationLevel(
        type_k="tq2_0", type_v="f16", label="tq2_0_keys",
        estimated_compression=15.52,
        metadata={"ggml_type_id": 35, "encoding": "2bit_polar"},
    ),
    KVQuantizationLevel(
        type_k="tq1_0", type_v="f16", label="tq1_0_keys",
        estimated_compression=18.96,
        metadata={"ggml_type_id": 34, "encoding": "ternary_base3"},
    ),
)


def next_kv_quantization_level(
    current_type_k: str = "f16",
) -> KVQuantizationLevel | None:
    """Return the next KV quantization level after the current type_k.

    Returns None when the progression is exhausted (caller should
    advance to the next fallback ladder step).
    """

    for i, level in enumerate(KV_QUANTIZATION_PROGRESSION):
        if level.type_k == current_type_k:
            if i + 1 < len(KV_QUANTIZATION_PROGRESSION):
                return KV_QUANTIZATION_PROGRESSION[i + 1]
            return None
    return None


@dataclass(frozen=True, slots=True)
class OscillationControls:
    """Thresholds used to prevent fallback thrashing."""

    cooldown_seconds: int = 30
    minimum_dwell_seconds: int = 15
    churn_window_seconds: int = 30
    churn_threshold: int = 1


# Throughput-optimized timings: faster recovery on high-VRAM systems
THROUGHPUT_OSCILLATION_CONTROLS = OscillationControls(
    cooldown_seconds=10,
    minimum_dwell_seconds=5,
    churn_window_seconds=15,
    churn_threshold=1,
)

# Default stability timings (unchanged from v1)
STABILITY_OSCILLATION_CONTROLS = OscillationControls(
    cooldown_seconds=30,
    minimum_dwell_seconds=15,
    churn_window_seconds=30,
    churn_threshold=1,
)


@dataclass(frozen=True, slots=True)
class FallbackState:
    """Observed fallback state used by the control logic."""

    current_step: FallbackStep | None
    last_step_change_ms: int | None
    step_change_history_ms: tuple[int, ...] = ()
    stability_mode: bool = False


@dataclass(frozen=True, slots=True)
class FallbackDecision:
    """Deterministic result of evaluating the fallback policy."""

    should_advance: bool
    next_step: FallbackStep | None
    enter_stability_mode: bool
    reason_code: str


class FallbackController:
    """Ladder planner with anti-oscillation safeguards."""

    def __init__(
        self,
        ladder: Sequence[FallbackStep] | None = None,
        controls: OscillationControls | None = None,
    ) -> None:
        self._ladder = tuple(ladder or DEFAULT_FALLBACK_LADDER)
        self._controls = controls or OscillationControls()

    @property
    def ladder(self) -> tuple[FallbackStep, ...]:
        """Expose the fallback ladder in execution order."""

        return self._ladder

    @property
    def controls(self) -> OscillationControls:
        """Expose the oscillation controls used by the controller."""

        return self._controls

    def next_step(self, current_step: FallbackStep | None) -> FallbackStep | None:
        """Return the next ladder step after the current one."""

        if current_step is None:
            # M12 fix: guard against empty ladder
            return self._ladder[0] if self._ladder else None

        try:
            index = self._ladder.index(current_step)
        except ValueError:
            return None

        if index + 1 >= len(self._ladder):
            return None
        return self._ladder[index + 1]

    def evaluate(
        self,
        state: FallbackState,
        now_ms: int,
    ) -> FallbackDecision:
        """Evaluate whether the ladder may advance at the current timestamp."""

        if state.current_step is None:
            return FallbackDecision(
                should_advance=True,
                next_step=self._ladder[0],
                enter_stability_mode=False,
                reason_code="FALLBACK_INITIAL_STEP",
            )

        if self._is_churn_triggered(state.step_change_history_ms, now_ms):
            return FallbackDecision(
                should_advance=False,
                next_step=state.current_step,
                enter_stability_mode=True,
                reason_code="FALLBACK_CHURN_TRIGGERED_STABILITY_MODE",
            )

        if state.last_step_change_ms is not None:
            elapsed_ms = now_ms - state.last_step_change_ms
            cooldown_ms = self._controls.cooldown_seconds * 1000
            dwell_ms = self._controls.minimum_dwell_seconds * 1000
            if elapsed_ms < cooldown_ms:
                return FallbackDecision(
                    should_advance=False,
                    next_step=state.current_step,
                    enter_stability_mode=state.stability_mode,
                    reason_code="FALLBACK_COOLDOWN_ACTIVE",
                )
            if elapsed_ms < dwell_ms:
                return FallbackDecision(
                    should_advance=False,
                    next_step=state.current_step,
                    enter_stability_mode=state.stability_mode,
                    reason_code="FALLBACK_MINIMUM_DWELL_ACTIVE",
                )

        next_step = self.next_step(state.current_step)
        if next_step is None:
            return FallbackDecision(
                should_advance=False,
                next_step=state.current_step,
                enter_stability_mode=state.stability_mode,
                reason_code="FALLBACK_FINAL_STEP_REACHED",
            )

        return FallbackDecision(
            should_advance=True,
            next_step=next_step,
            enter_stability_mode=state.stability_mode,
            reason_code="FALLBACK_ADVANCE_ALLOWED",
        )

    def _is_churn_triggered(
        self,
        step_change_history_ms: Sequence[int],
        now_ms: int,
    ) -> bool:
        """Detect whether the recent step-change rate exceeds the policy threshold."""

        window_ms = self._controls.churn_window_seconds * 1000
        recent_changes = [
            timestamp
            for timestamp in step_change_history_ms
            if now_ms - timestamp <= window_ms
        ]
        return len(recent_changes) > self._controls.churn_threshold

