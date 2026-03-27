"""Phase 3 service-host runtime tests."""

from __future__ import annotations

import time
import unittest

from astrawave.errors import ApiError, ApiErrorCode
from astrawave.ipc_server import AstraWeaveIpcServer

try:  # pragma: no cover - optional Phase 3 host module
    from astrawave.service_host import AstraWeaveServiceHost, ServiceHostConfig
except ImportError:  # pragma: no cover - current repo falls back to the IPC server
    AstraWeaveServiceHost = None
    ServiceHostConfig = None


from tests.conftest import endpoint_to_uri


def _endpoint_uri(endpoint: object) -> str:
    return endpoint_to_uri(endpoint)


class _RuntimeHarness:
    """Use a Phase 3 host if present, otherwise exercise the IPC server directly."""

    def __init__(self) -> None:
        if AstraWeaveServiceHost is not None:
            self.runtime = AstraWeaveServiceHost(
                ServiceHostConfig(
                    endpoint="tcp://127.0.0.1:0",
                    prefer_named_pipe=False,
                    host="127.0.0.1",
                    port=0,
                )
            )
        else:
            self.runtime = AstraWeaveIpcServer(prefer_named_pipe=False, host="127.0.0.1", port=0)

    def start(self) -> None:
        self.runtime.start()

    def stop(self) -> None:
        self.runtime.stop()

    def run_for(self, duration_seconds: float) -> None:
        self.start()
        try:
            time.sleep(duration_seconds)
        finally:
            self.stop()

    @property
    def endpoint(self) -> object:
        status = self.status()
        return status["endpoint"] if isinstance(status, dict) else status.endpoint

    @property
    def transport(self) -> object:
        status = self.status()
        return status["transport"] if isinstance(status, dict) else status.transport

    @property
    def is_running(self) -> bool:
        return self.runtime.is_running

    def status(self) -> dict[str, object]:
        if hasattr(self.runtime, "status"):
            status = self.runtime.status()
            if isinstance(status, dict):
                return status
            return {
                "running": status.running,
                "endpoint": status.endpoint,
                "transport": status.transport,
                "uptime_seconds": status.uptime_seconds,
                "served_connections": status.served_connections,
                "served_requests": status.served_requests,
            }
        return {
            "running": self.is_running,
            "endpoint": self.endpoint,
            "transport": self.transport,
        }


class Phase3ServiceHostTests(unittest.TestCase):
    def test_start_exposes_usable_endpoint(self) -> None:
        harness = _RuntimeHarness()
        self.addCleanup(harness.stop)

        harness.start()
        self.assertTrue(harness.is_running)

        endpoint = harness.endpoint
        self.assertIsNotNone(endpoint)
        self.assertIn(harness.transport, {"localhost", "named_pipe", None})

        endpoint_uri = _endpoint_uri(endpoint)
        self.assertTrue(endpoint_uri.startswith(("tcp://", "pipe://")))

        status = harness.status()
        self.assertIn("endpoint", status)
        self.assertIn("transport", status)
        self.assertTrue(status["running"])

    def test_controlled_runtime_run_for_shuts_down_cleanly(self) -> None:
        harness = _RuntimeHarness()
        harness.run_for(0.05)

        self.assertFalse(harness.is_running)
        status = harness.status()
        self.assertFalse(status["running"])
        self.assertIsNotNone(status["endpoint"])

    def test_non_localhost_tcp_bind_is_rejected(self) -> None:
        if AstraWeaveServiceHost is not None and ServiceHostConfig is not None:
            host = AstraWeaveServiceHost(
                ServiceHostConfig(
                    endpoint="tcp://0.0.0.0:0",
                    prefer_named_pipe=False,
                    host="127.0.0.1",
                    port=0,
                )
            )
            with self.assertRaises(ValueError):
                host.start()
            return

        server = AstraWeaveIpcServer(prefer_named_pipe=False, host="0.0.0.0", port=0)
        with self.assertRaises(ApiError) as ctx:
            server.start()
        self.assertEqual(ctx.exception.code, ApiErrorCode.INVALID_ARGUMENT)


if __name__ == "__main__":
    unittest.main()
