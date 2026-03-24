"""Contract tests for the core AstraWeave service lifecycle."""

from __future__ import annotations

import unittest

from astrawave import AstraWeaveService
from astrawave.errors import ApiError, ApiErrorCode
from astrawave.types import MemoryTier


class AstraWeaveServiceLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = AstraWeaveService()
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


if __name__ == "__main__":
    unittest.main()
