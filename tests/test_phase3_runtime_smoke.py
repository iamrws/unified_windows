"""Phase 3 runtime smoke tests over the real IPC path."""

from __future__ import annotations

import os
import time
import unittest

from astrawave.errors import ApiError, ApiErrorCode
from astrawave.ipc_client import AstraWeaveIpcClient
from astrawave.ipc_server import AstraWeaveIpcServer
from astrawave.security import CallerIdentity, resolve_process_user_sid
from astrawave.types import MemoryTier


from tests.conftest import endpoint_to_uri, wait_for_server


def _endpoint_uri(endpoint: object) -> str:
    return endpoint_to_uri(endpoint)


def _wait_for_endpoint(server: AstraWeaveIpcServer) -> str:
    return wait_for_server(server)


class Phase3RuntimeSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = AstraWeaveIpcServer(
            prefer_named_pipe=False,
            host="127.0.0.1",
            port=0,
        )
        self.server.start()
        self.addCleanup(self.server.stop)

        self.owner_sid = self.server.service.security_guard.policy.service_owner_sid
        self.owner = CallerIdentity(user_sid=self.owner_sid, pid=os.getpid())
        self.foreign = CallerIdentity(
            user_sid=(resolve_process_user_sid(os.getpid()) or "S-1-5-21-foreign") + "-foreign",
            pid=os.getpid(),
        )
        self.endpoint = _wait_for_endpoint(self.server)

    def _make_client(self, caller: CallerIdentity) -> AstraWeaveIpcClient:
        client = AstraWeaveIpcClient(
            endpoint=self.endpoint,
            default_caller=caller,
            timeout=5.0,
            prefer_named_pipe=False,
        ).connect()
        self.addCleanup(client.close)
        return client

    def test_real_ipc_client_executes_basic_lifecycle_flow(self) -> None:
        client = self._make_client(self.owner)

        session_id = client.CreateSession(self.owner)
        self.assertIsInstance(session_id, str)
        client.LoadModel(session_id, "demo-model", self.owner, runtime_backend="ollama")
        client.RegisterTensor(session_id, "kv", 1024, self.owner)
        client.SetTierHint(session_id, "kv", MemoryTier.HOT, self.owner)

        prefetch_plan = client.PrefetchPlan(session_id, self.owner)
        self.assertIsInstance(prefetch_plan, list)
        self.assertEqual(len(prefetch_plan), 1)

        result = client.RunStep(
            session_id,
            "decode",
            self.owner,
            prompt="hello",
            max_tokens=8,
            temperature=0.1,
        )
        self.assertEqual(result["session_id"], session_id)
        self.assertEqual(result["step_name"], "decode")
        self.assertIn("correlation_id", result)

        residency = client.GetResidency(session_id, self.owner)
        pressure = client.GetPressure(session_id, self.owner)
        self.assertEqual(residency.session_id, session_id)
        self.assertEqual(pressure.session_id, session_id)

        client.CloseSession(session_id, self.owner)

        with self.assertRaises(ApiError) as cm:
            client.LoadModel(session_id, "another-model", self.owner)
        self.assertIn(cm.exception.code, {ApiErrorCode.INVALID_STATE, ApiErrorCode.NOT_FOUND})

    def test_remote_call_smoke_path_denies_foreign_caller(self) -> None:
        client = self._make_client(self.owner)
        session_id = client.CreateSession(self.owner)
        client.LoadModel(session_id, "demo-model", self.owner)

        with self.assertRaises(ApiError) as cm:
            client.GetResidency(session_id, self.foreign)
        self.assertEqual(cm.exception.code, ApiErrorCode.AUTH_DENIED)

        client.CloseSession(session_id, self.owner)


if __name__ == "__main__":
    unittest.main()
