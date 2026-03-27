"""Contract tests for fallback ordering and anti-oscillation behavior."""

from __future__ import annotations

import importlib
import unittest


class FallbackContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.fallback = importlib.import_module("astrawave.fallback")
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise AssertionError("astrawave.fallback is required for the fallback contract tests") from exc

        try:
            cls.FallbackController = getattr(cls.fallback, "FallbackController")
            cls.FallbackStep = getattr(cls.fallback, "FallbackStep")
        except AttributeError as exc:  # pragma: no cover
            raise AssertionError("astrawave.fallback must expose FallbackController and FallbackStep") from exc

    def test_fallback_ladder_order_is_mandatory(self) -> None:
        expected = [
            "KV_QUANTIZATION_UPGRADE",
            "KV_CONTEXT_REDUCTION",
            "BATCH_REDUCTION",
            "PRECISION_REDUCTION",
            "SELECTIVE_CPU_OFFLOAD",
            "CONTROLLED_FAIL",
        ]

        actual = [step.name for step in self.FallbackStep]
        self.assertEqual(actual, expected)

    def test_anti_oscillation_defaults_are_locked(self) -> None:
        controller = self.FallbackController()

        self.assertEqual(controller.controls.cooldown_seconds, 15)
        self.assertEqual(controller.controls.minimum_dwell_seconds, 10)
        self.assertEqual(controller.controls.churn_threshold, 1)

    def test_controller_declines_step_changes_inside_cooldown_window(self) -> None:
        controller = self.FallbackController()
        state = self.fallback.FallbackState(
            current_step=self.FallbackStep.BATCH_REDUCTION,
            last_step_change_ms=0,
            step_change_history_ms=(0, 1000),
            stability_mode=False,
        )

        decision = controller.evaluate(state, 10_000)
        self.assertFalse(decision.should_advance)
        self.assertTrue(decision.enter_stability_mode)
        self.assertEqual(decision.next_step, self.FallbackStep.BATCH_REDUCTION)
        self.assertEqual(decision.reason_code, "FALLBACK_CHURN_TRIGGERED_STABILITY_MODE")

    def test_controller_respects_cooldown_before_advancing(self) -> None:
        controller = self.FallbackController()
        state = self.fallback.FallbackState(
            current_step=self.FallbackStep.BATCH_REDUCTION,
            last_step_change_ms=0,
            step_change_history_ms=(),
            stability_mode=False,
        )

        decision = controller.evaluate(state, 10_000)
        self.assertFalse(decision.should_advance)
        self.assertEqual(decision.reason_code, "FALLBACK_COOLDOWN_ACTIVE")


if __name__ == "__main__":
    unittest.main()
