"""Phase 2 end-to-end stress test for the IPC-backed AstraWeave stack."""

from __future__ import annotations

from contextlib import suppress
import os
import unittest

from astrawave.errors import ApiError, ApiErrorCode
from astrawave.ipc_client import AstraWeaveIpcClient
from astrawave.ipc_server import AstraWeaveIpcServer
from astrawave.sdk import AstraWeaveSDK
from astrawave.security import CallerIdentity, resolve_process_user_sid
from astrawave.telemetry import TelemetryReasonCode
from astrawave.types import MemoryTier


class Phase2EndToEndTests(unittest.TestCase):
    """Exercise the real IPC path with caller enforcement and telemetry flow."""

    def setUp(self) -> None:
        self.server = AstraWeaveIpcServer(
            prefer_named_pipe=False,
            host="127.0.0.1",
            port=0,
        )
        self.server.start()
        self.addCleanup(self._cleanup)

        self.owner_sid = self.server.service.security_guard.policy.service_owner_sid
        self.owner = CallerIdentity(user_sid=self.owner_sid, pid=os.getpid())
        self.foreign = CallerIdentity(
            user_sid=(resolve_process_user_sid(os.getpid()) or "S-1-5-21-foreign") + "-foreign",
            pid=os.getpid(),
        )

        endpoint = self._endpoint_uri(self.server.endpoint)
        self.client = AstraWeaveIpcClient(
            endpoint=endpoint,
            default_caller=self.owner,
            timeout=5.0,
            prefer_named_pipe=False,
        ).connect()
        self.sdk = AstraWeaveSDK(client=self.client)

    def _cleanup(self) -> None:
        for obj in (getattr(self, "sdk", None), getattr(self, "client", None)):
            if obj is None:
                continue
            close = getattr(obj, "close", None)
            if callable(close):
                with suppress(Exception):
                    close()
        server = getattr(self, "server", None)
        if server is not None:
            with suppress(Exception):
                server.stop()

    @staticmethod
    def _endpoint_uri(endpoint: object) -> str:
        from tests.conftest import endpoint_to_uri
        return endpoint_to_uri(endpoint)

    def test_owner_flow_emits_telemetry_and_remote_state(self) -> None:
        session_id = self.sdk.CreateSession()
        self.sdk.LoadModel(session_id, "demo-model", runtime_backend="ollama")
        self.sdk.RegisterTensor(session_id, "kv", 1)
        self.sdk.SetTierHint(session_id, "kv", MemoryTier.HOT)

        session = self.server.service._sessions[session_id]
        with session.lock:
            session.vram_budget_bytes = 1

        prefetch_plan = self.sdk.PrefetchPlan(session_id)
        self.assertEqual(len(prefetch_plan), 1)
        self.assertEqual(prefetch_plan[0]["reason_code"], "prefetch")
        self.assertEqual(prefetch_plan[0]["destination"], "VRAM")

        pressure = self.sdk.GetPressure(session_id)
        self.assertEqual(pressure.session_id, session_id)
        self.assertAlmostEqual(pressure.pressure_level, 1.0)

        run_result = self.sdk.RunStep(
            session_id,
            step_name="decode",
            prompt="hello",
            max_tokens=8,
            temperature=0.1,
        )
        self.assertEqual(run_result["session_id"], session_id)
        self.assertEqual(run_result["step_name"], "decode")
        self.assertEqual(run_result["state"], "DEGRADED")
        self.assertEqual(run_result["fallback_result"]["next_step"], "kv_quantization_upgrade")

        residency = self.sdk.GetResidency(session_id)
        self.assertEqual(residency.session_state.value, "DEGRADED")
        # Phase 7: first fallback step is KV_QUANTIZATION_UPGRADE which
        # compresses in-place (no demotion), so tensor stays in VRAM/HOT.
        self.assertEqual(residency.primary_tier.value, "HOT")
        self.assertEqual(residency.tensor_residency["kv"].value, "VRAM")

        with self.assertRaises(ApiError) as cm:
            self.client.GetResidency(session_id, caller=self.foreign)
        self.assertEqual(cm.exception.code, ApiErrorCode.AUTH_DENIED)

        telemetry_records = self.server.service.telemetry.records
        reason_codes = [record.reason_code for record in telemetry_records]
        self.assertGreaterEqual(len(telemetry_records), 4)
        self.assertIn(TelemetryReasonCode.UNKNOWN, reason_codes)
        self.assertIn(TelemetryReasonCode.TRANSFER_PREFETCH, reason_codes)
        self.assertIn(TelemetryReasonCode.FALLBACK_KV_QUANTIZATION_UPGRADE, reason_codes)
        for record in telemetry_records:
            self.assertTrue(record.correlation_id.startswith("tel-"))
            self.assertIsNotNone(record.session_id_hash)
            self.assertNotEqual(record.session_id_hash, session_id)

    def test_remote_close_is_idempotent_and_post_close_access_fails(self) -> None:
        session_id = self.sdk.CreateSession()
        self.sdk.LoadModel(session_id, "demo-model")

        self.sdk.CloseSession(session_id)
        self.sdk.CloseSession(session_id)

        with self.assertRaises(ApiError) as cm:
            self.sdk.LoadModel(session_id, "another-model")
        self.assertIn(cm.exception.code, {ApiErrorCode.INVALID_STATE, ApiErrorCode.NOT_FOUND})


if __name__ == "__main__":
    unittest.main()
