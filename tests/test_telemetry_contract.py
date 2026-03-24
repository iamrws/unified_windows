"""Contract tests for telemetry defaults, privacy, and export controls."""

from __future__ import annotations

import importlib
from inspect import signature
import unittest

from astrawave.errors import ApiError, ApiErrorCode


class TelemetryContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.telemetry = importlib.import_module("astrawave.telemetry")
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise AssertionError("astrawave.telemetry is required for the telemetry contract tests") from exc

        try:
            cls.TelemetryPolicy = getattr(cls.telemetry, "TelemetryPolicy")
            cls.TelemetryPipeline = getattr(cls.telemetry, "TelemetryPipeline")
        except AttributeError as exc:  # pragma: no cover
            raise AssertionError(
                "astrawave.telemetry must expose TelemetryPolicy and TelemetryPipeline"
            ) from exc

    def _make_policy(self, **overrides):
        params = signature(self.TelemetryPolicy).parameters
        filtered = {name: value for name, value in overrides.items() if name in params}
        return self.TelemetryPolicy(**filtered)

    def test_local_only_is_default_and_export_is_disabled(self) -> None:
        policy = self.TelemetryPolicy()

        self.assertTrue(getattr(policy, "local_only", False))
        self.assertFalse(getattr(policy, "export_opt_in", True))

        pipeline = self.TelemetryPipeline(policy)
        recorded_policy = getattr(pipeline, "policy", policy)

        self.assertTrue(getattr(recorded_policy, "local_only", False))
        self.assertFalse(getattr(recorded_policy, "export_opt_in", True))

    def test_export_requires_explicit_opt_in(self) -> None:
        policy = self._make_policy(local_only=True, export_opt_in=False)
        pipeline = self.TelemetryPipeline(policy)

        with self.assertRaises(ApiError) as cm:
            pipeline.build_export_bundle()
        self.assertEqual(cm.exception.code, ApiErrorCode.INVALID_STATE)

        opt_in_policy = self._make_policy(local_only=True, export_opt_in=True)
        opt_in_pipeline = self.TelemetryPipeline(opt_in_policy)
        event = self.telemetry.TelemetryEvent(
            reason_code=self.telemetry.TelemetryReasonCode.POLICY_CHANGED,
            session_id="session-1",
        )
        opt_in_pipeline.record_event(event)
        bundle = opt_in_pipeline.build_export_bundle()
        self.assertIsNotNone(bundle)
        self.assertTrue(bundle.to_dict()["policy"]["local_only"])
        self.assertTrue(bundle.to_dict()["policy"]["export_opt_in"])


if __name__ == "__main__":
    unittest.main()
