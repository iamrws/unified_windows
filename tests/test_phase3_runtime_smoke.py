"""Phase 3 runtime smoke tests over the real IPC path."""

from __future__ import annotations

import time
import unittest

from astrawave.errors import ApiError, ApiErrorCode
from astrawave.ipc_client import AstraWeaveIpcClient
from astrawave.ipc_server import AstraWeaveIpcServer
from astrawave.security import CallerIdentity
from astrawave.types import MemoryTier


def _endpoint_uri(endpoint: object) -> str:
    if isinstance(endpoint, tuple) and len(endpoint) == 2:
        host, port = endpoint
        return f"tcp://{host}:{port}"
    if isinstance(endpoint, str):
        if endpoint.startswith("\\\\.\\pipe\\"):
            return f"pipe://{endpoint}"
        if endpoint.startswith(("tcp://", "pipe://")):
            return endpoint
        return f"tcp://{endpoint}"
    raise AssertionError(f"Unsupported endpoint shape: {endpoint!r}")


def _wait_for_endpoint(server: AstraWeaveIpcServer) -> str:
    last_endpoint: object | None = None
    for _ in range(40):
        last_endpoint = server.endpoint
        if last_endpoint:
            return _endpoint_uri(last_endpoint)
        time.sleep(0.05)
    raise AssertionError(f"server did not expose an endpoint; last value={last_endpoint!r}")


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
        self.owner = CallerIdentity(user_sid=self.owner_sid, pid=1001)
        self.foreign = CallerIdentity(user_sid="S-1-5-21-2000", pid=2002)
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
        client.LoadModel(session_id, "demo-model", self.owner)
        client.RegisterTensor(session_id, "kv", 1024, self.owner)
        client.SetTierHint(session_id, "kv", MemoryTier.HOT, self.owner)

        prefetch_plan = client.PrefetchPlan(session_id, self.owner)
        self.assertIsInstance(prefetch_plan, list)
        self.assertEqual(len(prefetch_plan), 1)

        result = client.RunStep(session_id, "decode", self.owner)
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
