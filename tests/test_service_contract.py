"""Contract tests for the core AstraWeave service lifecycle."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from astrawave import AstraWeaveService
from astrawave.errors import ApiError, ApiErrorCode
from astrawave.types import MemoryTier


class AstraWeaveServiceLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = AstraWeaveService(runstep_mode="simulation")
        self.session_id = self.service.CreateSession()

    def assertApiErrorCode(self, exc: Exception, expected: ApiErrorCode) -> None:
        self.assertIsInstance(exc, ApiError)
        self.assertEqual(exc.code, expected)

    def test_invalid_order_rejection_blocks_out_of_sequence_calls(self) -> None:
        with self.assertRaises(ApiError) as cm:
            self.service.RunStep(self.session_id)
        self.assertApiErrorCode(cm.exception, ApiErrorCode.INVALID_STATE)

        with self.assertRaises(ApiError) as cm:
            self.service.RegisterTensor(self.session_id, "kv", 1024)
        self.assertApiErrorCode(cm.exception, ApiErrorCode.INVALID_STATE)

        with self.assertRaises(ApiError) as cm:
            self.service.SetTierHint(self.session_id, "kv", MemoryTier.HOT)
        self.assertApiErrorCode(cm.exception, ApiErrorCode.INVALID_STATE)

        with self.assertRaises(ApiError) as cm:
            self.service.PrefetchPlan(self.session_id)
        self.assertApiErrorCode(cm.exception, ApiErrorCode.INVALID_STATE)

        self.service.LoadModel(self.session_id, "demo-model")
        self.service.RegisterTensor(self.session_id, "kv", 1024)
        self.service.SetTierHint(self.session_id, "kv", MemoryTier.WARM)

        result = self.service.RunStep(self.session_id, step_name="decode")
        self.assertEqual(result["session_id"], self.session_id)
        self.assertEqual(result["step_name"], "decode")

    def test_close_session_is_idempotent(self) -> None:
        self.service.LoadModel(self.session_id, "demo-model")
        self.service.CloseSession(self.session_id)

        # Second close must be a no-op, not an exception.
        self.service.CloseSession(self.session_id)

        with self.assertRaises(ApiError) as cm:
            self.service.LoadModel(self.session_id, "another-model")
        self.assertIn(cm.exception.code, {ApiErrorCode.INVALID_STATE, ApiErrorCode.NOT_FOUND})

    def test_register_tensor_after_load_is_allowed(self) -> None:
        self.service.LoadModel(self.session_id, "demo-model")
        self.service.RegisterTensor(self.session_id, "weights", 4096)

        residency = self.service.GetResidency(self.session_id)
        self.assertEqual(residency.session_id, self.session_id)
        self.assertGreaterEqual(residency.pinned_ram_bytes, 4096)

    def test_run_step_rejects_second_active_execution(self) -> None:
        self.service.LoadModel(self.session_id, "demo-model")
        self.service.RegisterTensor(self.session_id, "kv", 1024)
        session = self.service._get_session(self.session_id)  # contract-level active-run guard check
        with session.lock:
            session.active_run = True
        try:
            with self.assertRaises(ApiError) as cm:
                self.service.RunStep(self.session_id, step_name="decode")
            self.assertApiErrorCode(cm.exception, ApiErrorCode.CONFLICT_RUN_IN_PROGRESS)
        finally:
            with session.lock:
                session.active_run = False

    def test_run_step_hardware_mode_invokes_executor_and_returns_hardware_payload(self) -> None:
        calls: list[dict[str, int]] = []

        def fake_executor(*, size_bytes: int, device_index: int, hold_ms: int) -> dict[str, object]:
            calls.append(
                {
                    "size_bytes": size_bytes,
                    "device_index": device_index,
                    "hold_ms": hold_ms,
                }
            )
            return {
                "ok": True,
                "nvml_observation": {
                    "observed": True,
                    "observed_delta_bytes": 4096,
                },
            }

        service = AstraWeaveService(runstep_mode="hardware", hardware_executor=fake_executor)
        session_id = service.CreateSession()
        service.LoadModel(session_id, "demo-model")
        service.RegisterTensor(session_id, "kv", 1024)

        result = service.RunStep(session_id, step_name="decode")
        self.assertEqual(result["requested_run_mode"], "hardware")
        self.assertEqual(result["run_mode"], "hardware")
        self.assertIsInstance(result["hardware_result"], dict)
        self.assertTrue(result["hardware_result"]["ok"])
        self.assertEqual(len(calls), 1)
        self.assertGreaterEqual(calls[0]["size_bytes"], 1024)

    def test_run_step_hardware_mode_marks_session_degraded_on_runtime_failure(self) -> None:
        def failing_executor(*, size_bytes: int, device_index: int, hold_ms: int) -> dict[str, object]:
            return {
                "ok": False,
                "error": {"code": "CUDA_POC_DRIVER_MISSING", "message": "nvcuda.dll was not found"},
            }

        service = AstraWeaveService(runstep_mode="hardware", hardware_executor=failing_executor)
        session_id = service.CreateSession()
        service.LoadModel(session_id, "demo-model")
        service.RegisterTensor(session_id, "kv", 2048)

        result = service.RunStep(session_id, step_name="decode")
        self.assertEqual(result["requested_run_mode"], "hardware")
        self.assertEqual(result["run_mode"], "simulation")
        self.assertIsInstance(result["hardware_result"], dict)
        self.assertFalse(result["hardware_result"]["ok"])
        self.assertTrue(result["hardware_result"]["fallback_to_simulation"])
        self.assertEqual(result["state"].value, "DEGRADED")

    def test_env_flag_can_enable_hardware_mode_without_constructor_override(self) -> None:
        def fake_executor(*, size_bytes: int, device_index: int, hold_ms: int) -> dict[str, object]:
            return {"ok": True}

        with patch.dict(os.environ, {"ASTRAWEAVE_ENABLE_HARDWARE_RUNSTEP": "1"}, clear=False):
            service = AstraWeaveService(hardware_executor=fake_executor)
            session_id = service.CreateSession()
            service.LoadModel(session_id, "demo-model")
            service.RegisterTensor(session_id, "kv", 4096)
            result = service.RunStep(session_id, step_name="decode")
            self.assertEqual(result["requested_run_mode"], "hardware")
            self.assertEqual(result["run_mode"], "hardware")

    def test_auto_mode_prefers_hardware_and_falls_back_cleanly(self) -> None:
        def failing_executor(*, size_bytes: int, device_index: int, hold_ms: int) -> dict[str, object]:
            return {
                "ok": False,
                "error": {"code": "CUDA_POC_DRIVER_MISSING", "message": "nvcuda.dll was not found"},
            }

        service = AstraWeaveService(hardware_executor=failing_executor)
        session_id = service.CreateSession()
        service.LoadModel(session_id, "demo-model")
        service.RegisterTensor(session_id, "kv", 2048)

        result = service.RunStep(session_id, step_name="decode")
        self.assertEqual(result["requested_run_mode"], "auto")
        self.assertEqual(result["run_mode"], "simulation")
        self.assertFalse(result["hardware_result"]["ok"])
        self.assertTrue(result["hardware_result"]["fallback_to_simulation"])
        self.assertEqual(result["state"].value, "DEGRADED")


if __name__ == "__main__":
    unittest.main()
