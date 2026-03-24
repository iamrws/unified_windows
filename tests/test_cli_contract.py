"""Contract tests for the AstraWeave CLI."""

from __future__ import annotations

import json
import os
import hashlib
import subprocess
import sys
import unittest
from pathlib import Path
import tempfile


class AstraWeaveCliContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.endpoint = "local://default"
        self.python = sys.executable
        self.repo_root = Path(__file__).resolve().parents[1]
        self.state_path = self._endpoint_path(self.endpoint)
        self._cleanup_state()

    def tearDown(self) -> None:
        self._cleanup_state()

    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.repo_root)
        return subprocess.run(
            [self.python, "-m", "astrawave.cli", "--endpoint", self.endpoint, *args],
            cwd=self.repo_root,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def run_cli_default(self, *args: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.repo_root)
        return subprocess.run(
            [self.python, "-m", "astrawave.cli", *args],
            cwd=self.repo_root,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def _endpoint_path(self, endpoint: str) -> Path:
        digest = hashlib.sha256(endpoint.encode("utf-8")).hexdigest()
        return Path(tempfile.gettempdir()) / "astrawave_cli_state" / f"{digest}.json"

    def _cleanup_state(self) -> None:
        if self.state_path.exists():
            self.state_path.unlink()

    def parse_json(self, payload: str) -> dict[str, object]:
        data = json.loads(payload)
        self.assertIsInstance(data, dict)
        return data

    def assert_success_envelope(self, payload: str) -> dict[str, object]:
        data = self.parse_json(payload)
        self.assertTrue(data["ok"])
        self.assertIn("result", data)
        return data

    def assert_error_envelope(self, payload: str) -> dict[str, object]:
        data = self.parse_json(payload)
        self.assertFalse(data["ok"])
        self.assertIn("error", data)
        error = data["error"]
        self.assertIsInstance(error, dict)
        self.assertIn("code", error)
        self.assertIn("message", error)
        return data

    def test_success_output_is_json_envelope(self) -> None:
        completed = self.run_cli("create-session")
        self.assertEqual(completed.returncode, 0, completed.stderr)

        envelope = self.assert_success_envelope(completed.stdout)
        result = envelope["result"]
        self.assertIsInstance(result, dict)
        self.assertIn("session_id", result)

    def test_error_output_is_json_envelope_and_nonzero_exit(self) -> None:
        completed = self.run_cli("load-model", "missing-session", "demo-model")
        self.assertNotEqual(completed.returncode, 0)

        envelope = self.assert_error_envelope(completed.stderr)
        error = envelope["error"]
        self.assertEqual(error["code"], "AW_ERR_NOT_FOUND")

    def test_local_backend_snapshot_mode_supports_multi_command_flow(self) -> None:
        create = self.run_cli("create-session")
        self.assertEqual(create.returncode, 0, create.stderr)
        create_envelope = self.assert_success_envelope(create.stdout)
        session_id = create_envelope["result"]["session_id"]

        load = self.run_cli("load-model", str(session_id), "demo-model")
        self.assertEqual(load.returncode, 0, load.stderr)
        self.assert_success_envelope(load.stdout)

        register = self.run_cli("register-tensor", str(session_id), "kv", "1024")
        self.assertEqual(register.returncode, 0, register.stderr)
        self.assert_success_envelope(register.stdout)

        tier = self.run_cli("set-tier-hint", str(session_id), "kv", "HOT")
        self.assertEqual(tier.returncode, 0, tier.stderr)
        self.assert_success_envelope(tier.stdout)

        prefetch = self.run_cli("prefetch-plan", str(session_id))
        self.assertEqual(prefetch.returncode, 0, prefetch.stderr)
        prefetch_envelope = self.assert_success_envelope(prefetch.stdout)
        self.assertIn("migrations", prefetch_envelope["result"])

        step = self.run_cli("run-step", str(session_id), "--step-name", "decode")
        self.assertEqual(step.returncode, 0, step.stderr)
        step_envelope = self.assert_success_envelope(step.stdout)
        self.assertEqual(step_envelope["result"]["session_id"], session_id)

        residency = self.run_cli("get-residency", str(session_id))
        self.assertEqual(residency.returncode, 0, residency.stderr)
        residency_envelope = self.assert_success_envelope(residency.stdout)
        self.assertEqual(residency_envelope["result"]["session_id"], session_id)

        close = self.run_cli("close-session", str(session_id))
        self.assertEqual(close.returncode, 0, close.stderr)
        close_envelope = self.assert_success_envelope(close.stdout)
        self.assertEqual(close_envelope["result"]["closed"], True)

    def test_default_backend_prefers_remote_and_fails_cleanly_without_server(self) -> None:
        auto_state = self._endpoint_path("auto")
        if auto_state.exists():
            auto_state.unlink()
        self.addCleanup(lambda: auto_state.exists() and auto_state.unlink())

        completed = self.run_cli_default("create-session")
        self.assertNotEqual(completed.returncode, 0)
        envelope = self.assert_error_envelope(completed.stderr)
        self.assertEqual(envelope["error"]["code"], "AW_ERR_INTERNAL")
        self.assertFalse(auto_state.exists())

    def test_local_backend_can_be_selected_explicitly(self) -> None:
        completed = self.run_cli_default("--backend", "local", "create-session")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        envelope = self.assert_success_envelope(completed.stdout)
        self.assertIn("session_id", envelope["result"])

    def test_invalid_tcp_endpoint_is_reported_as_typed_json_error(self) -> None:
        completed = self.run_cli_default("--endpoint", "tcp://127.0.0.1:notaport", "create-session")
        self.assertNotEqual(completed.returncode, 0)
        envelope = self.assert_error_envelope(completed.stderr)
        self.assertEqual(envelope["error"]["code"], "AW_ERR_INVALID_ARGUMENT")
        self.assertNotIn("Traceback", completed.stderr)

    def test_serve_rejects_non_localhost_tcp_bind(self) -> None:
        completed = self.run_cli_default(
            "serve",
            "--transport",
            "tcp",
            "--endpoint",
            "tcp://0.0.0.0:8765",
            "--duration-seconds",
            "0",
        )
        self.assertNotEqual(completed.returncode, 0)
        envelope = self.assert_error_envelope(completed.stderr)
        self.assertEqual(envelope["error"]["code"], "AW_ERR_INVALID_ARGUMENT")
        self.assertNotIn("Traceback", completed.stderr)

    def test_hardware_probe_returns_structured_json_even_without_nvidia_tools(self) -> None:
        completed = self.run_cli_default("hardware-probe")
        self.assertEqual(completed.returncode, 0, completed.stderr)

        envelope = self.assert_success_envelope(completed.stdout)
        result = envelope["result"]
        self.assertIsInstance(result, dict)
        self.assertIn("timestamp_ms", result)
        self.assertIsInstance(result["timestamp_ms"], int)
        self.assertIn("warnings", result)
        self.assertIsInstance(result["warnings"], list)
        self.assertIn("effective", result)
        self.assertIsInstance(result["effective"], dict)
        self.assertIn("nvidia_smi", result)
        self.assertIsInstance(result["nvidia_smi"], dict)
        self.assertIn("nvml", result)
        self.assertIsInstance(result["nvml"], dict)


if __name__ == "__main__":
    unittest.main()
