"""Integrated service contract tests for security and telemetry behavior."""

from __future__ import annotations

import unittest

from astrawave import AstraWeaveService
from astrawave.errors import ApiError, ApiErrorCode
from astrawave.security import CallerIdentity, SecurityGuard, SecurityPolicy
from astrawave.telemetry import (
    TelemetryPipeline,
    TelemetryPolicy,
    TelemetryReasonCode,
)
from astrawave.types import MemoryTier


class IntegratedServiceHarness:
    """Small test harness that composes service, security, and telemetry."""

    def __init__(
        self,
        *,
        owner_sid: str = "S-1-5-21-1000",
        create_session_limit_per_minute: int = 6,
        max_concurrent_sessions_per_caller: int = 8,
    ) -> None:
        self.guard = SecurityGuard(
            SecurityPolicy(
                service_owner_sid=owner_sid,
                create_session_limit_per_minute=create_session_limit_per_minute,
                max_concurrent_sessions_per_caller=max_concurrent_sessions_per_caller,
            )
        )
        self.telemetry = TelemetryPipeline(TelemetryPolicy())
        self.service = AstraWeaveService(
            security_guard=self.guard,
            telemetry_pipeline=self.telemetry,
        )
        self.session_owners: dict[str, str] = {}

    def create_session(self, caller: CallerIdentity) -> str:
        session_id = self.service.CreateSession(caller_identity=caller)
        self.session_owners[session_id] = caller.user_sid
        return session_id

    def load_model(self, caller: CallerIdentity, session_id: str, model_name: str) -> None:
        self.service.LoadModel(session_id, model_name, caller_identity=caller)

    def register_tensor(self, caller: CallerIdentity, session_id: str, tensor_name: str, size_bytes: int) -> None:
        self.service.RegisterTensor(session_id, tensor_name, size_bytes, caller_identity=caller)

    def run_step(self, caller: CallerIdentity, session_id: str, step_name: str = "run") -> dict[str, object]:
        return self.service.RunStep(session_id, step_name=step_name, caller_identity=caller)

    def get_residency(self, caller: CallerIdentity, session_id: str):
        return self.service.GetResidency(session_id, caller_identity=caller)

    def close_session(self, caller: CallerIdentity, session_id: str) -> None:
        self.service.CloseSession(session_id, caller_identity=caller)
        self.session_owners.pop(session_id, None)


class IntegratedServiceContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.owner = CallerIdentity(user_sid="S-1-5-21-1000", pid=1001)
        self.foreign = CallerIdentity(user_sid="S-1-5-21-2000", pid=2002)

    def test_session_ownership_isolation_denies_cross_user_access(self) -> None:
        harness = IntegratedServiceHarness()
        session_id = harness.create_session(self.owner)
        harness.load_model(self.owner, session_id, "demo-model")
        harness.register_tensor(self.owner, session_id, "kv", 1024)

        with self.assertRaises(ApiError) as cm:
            harness.get_residency(self.foreign, session_id)
        self.assertEqual(cm.exception.code, ApiErrorCode.AUTH_DENIED)

        with self.assertRaises(ApiError) as cm:
            harness.close_session(self.foreign, session_id)
        self.assertEqual(cm.exception.code, ApiErrorCode.AUTH_DENIED)

    def test_create_session_through_service_enforces_security_guard_limits(self) -> None:
        rate_limited_harness = IntegratedServiceHarness()
        for _ in range(6):
            rate_limited_harness.create_session(self.owner)

        with self.assertRaises(ApiError) as cm:
            rate_limited_harness.create_session(self.owner)
        self.assertEqual(cm.exception.code, ApiErrorCode.RATE_LIMITED)

        capped_harness = IntegratedServiceHarness(create_session_limit_per_minute=100)
        for _ in range(8):
            capped_harness.create_session(self.owner)

        with self.assertRaises(ApiError) as cm:
            capped_harness.create_session(self.owner)
        self.assertEqual(cm.exception.code, ApiErrorCode.RESOURCE_EXHAUSTED)

    def test_service_emits_telemetry_events_with_stable_reason_codes_and_correlation_ids(self) -> None:
        harness = IntegratedServiceHarness()
        session_id = harness.create_session(self.owner)
        harness.load_model(self.owner, session_id, "demo-model")
        harness.register_tensor(self.owner, session_id, "kv", 1024)
        harness.service.SetTierHint(session_id, "kv", MemoryTier.HOT, caller_identity=self.owner)
        harness.service.PrefetchPlan(session_id, caller_identity=self.owner)
        harness.run_step(self.owner, session_id, step_name="decode")
        harness.close_session(self.owner, session_id)

        records = harness.telemetry.records
        self.assertGreaterEqual(len(records), 3)

        reason_codes = [record.reason_code for record in records]
        self.assertIn(TelemetryReasonCode.UNKNOWN, reason_codes)
        self.assertIn(TelemetryReasonCode.TRANSFER_PREFETCH, reason_codes)

        for record in records:
            self.assertRegex(record.correlation_id, r"^tel-[0-9a-f-]{36}$")
            self.assertIsNotNone(record.session_id_hash)
            self.assertNotEqual(record.session_id_hash, session_id)

        event_types = {record.event_type.value for record in records}
        self.assertIn("custom", event_types)
        self.assertIn("transfer_event", event_types)


if __name__ == "__main__":
    unittest.main()
