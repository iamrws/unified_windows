"""Contract tests for release-gate readiness split reporting."""

from __future__ import annotations

import json
import shutil
import unittest
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from scripts.generate_release_gate_report import generate_reports


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class ReleaseGateReportContractTests(unittest.TestCase):
    @contextmanager
    def _workspace_tmpdir(self):
        base = Path("test_runtime")
        base.mkdir(parents=True, exist_ok=True)
        path = (base / f"release_gate_{uuid4().hex}").resolve()
        path.mkdir(parents=True, exist_ok=False)
        try:
            yield path
        finally:
            shutil.rmtree(path, ignore_errors=True)

    def _make_unittest_log(self, root: Path) -> Path:
        test_ids = sorted(
            {
                "test_security_contract.SecurityContractTests.test_unknown_or_cross_user_caller_is_denied_by_default",
                "test_ipc_server.IpcServerContractTests.test_server_requires_explicit_caller_identity_by_default",
                "test_ipc_client_sdk.IpcClientSdkE2ETests.test_connection_rejects_caller_identity_switch_after_binding",
                "test_integrated_service_contract.IntegratedServiceContractTests.test_session_ownership_isolation_denies_cross_user_access",
                "test_phase3_runtime_smoke.Phase3RuntimeSmokeTests.test_remote_call_smoke_path_denies_foreign_caller",
                "test_ipc_protocol.IpcProtocolContractTests.test_request_envelope_validation_rejects_malformed_input",
                "test_ipc_protocol.IpcProtocolContractTests.test_request_envelope_rejects_payloads_over_max_size",
                "test_ipc_server.IpcServerContractTests.test_server_rejects_oversized_request_payload",
                "test_telemetry_contract.TelemetryContractTests.test_local_only_is_default_and_export_is_disabled",
                "test_telemetry_contract.TelemetryContractTests.test_export_requires_explicit_opt_in",
                "test_capability_matrix_contract.CapabilityMatrixContractTests.test_profile_pa_primary_maps_to_numa_dgpu",
                "test_capability_matrix_contract.CapabilityMatrixContractTests.test_profile_pb_secondary_maps_to_cache_coherent_uma",
                "test_capability_matrix_contract.CapabilityMatrixContractTests.test_profile_pc_degraded_maps_to_uma",
                "test_capability_matrix_contract.CapabilityMatrixContractTests.test_profile_pd_unsupported_fails_fast",
                "test_fallback_contract.FallbackContractTests.test_controller_declines_step_changes_inside_cooldown_window",
                "test_fallback_contract.FallbackContractTests.test_controller_respects_cooldown_before_advancing",
                "test_fallback_contract.FallbackContractTests.test_anti_oscillation_defaults_are_locked",
                "test_integrated_service_contract.IntegratedServiceContractTests.test_service_emits_telemetry_events_with_stable_reason_codes_and_correlation_ids",
                "test_service_contract.AstraWeaveServiceLifecycleTests.test_invalid_order_rejection_blocks_out_of_sequence_calls",
                "test_service_contract.AstraWeaveServiceLifecycleTests.test_close_session_is_idempotent",
                "test_service_contract.AstraWeaveServiceLifecycleTests.test_run_step_rejects_second_active_execution",
                "test_ipc_protocol.IpcProtocolContractTests.test_error_envelope_maps_api_error_codes_stably",
            }
        )
        lines = [
            f"test_case_{index:03d} ({test_id}) ... ok"
            for index, test_id in enumerate(test_ids, start=1)
        ]
        lines.append("")
        lines.append(f"Ran {len(test_ids)} tests in 0.123s")
        lines.append("")
        lines.append("OK")
        path = root / "unittest.txt"
        _write(path, "\n".join(lines))
        return path

    def _make_inputs(self, root: Path) -> dict[str, Path]:
        unittest_log = self._make_unittest_log(root)
        w7 = root / "w7.json"
        w13 = root / "w13.json"
        operations = root / "ops.json"
        artifact_file = root / "placeholder.json"
        compliance_manifest = root / "compliance.json"

        _write(
            w7,
            json.dumps({"iterations": 100, "failure_count": 0, "run_step_p95_ms": 181.0}),
        )
        _write(
            w13,
            json.dumps({"iterations": 100, "failure_count": 0, "run_step_p95_ms": 183.0}),
        )
        _write(operations, json.dumps({"verdict": "pass"}))
        _write(artifact_file, json.dumps({"ok": True}))
        _write(
            compliance_manifest,
            json.dumps({"artifacts": {"placeholder": str(artifact_file)}}),
        )
        return {
            "unittest_log": unittest_log,
            "w7": w7,
            "w13": w13,
            "operations": operations,
            "compliance_manifest": compliance_manifest,
        }

    def test_generate_reports_marks_hardware_ready_when_hardware_gate_passes(self) -> None:
        with self._workspace_tmpdir() as root:
            inputs = self._make_inputs(root)
            out_dir = root / "release_gate"
            hardware_gate = out_dir / "hardware_gate_test.json"
            _write(
                hardware_gate,
                json.dumps(
                    {
                        "gate_checks": {
                            "service_hardware_mode_active": {"passed": True},
                            "service_hardware_transfer_ok": {"passed": True},
                            "service_triggered_nvml_delta_observed": {"passed": True},
                        }
                    }
                ),
            )

            result = generate_reports(
                run_id="test",
                out_dir=out_dir,
                unittest_log=inputs["unittest_log"],
                w7_summary=inputs["w7"],
                w13_summary=inputs["w13"],
                operations_report=inputs["operations"],
                compliance_manifest=inputs["compliance_manifest"],
                hardware_gate_report=hardware_gate,
            )
            self.assertEqual(result["verdict"], "hardware_ready")
            self.assertTrue(result["simulation_ready"])
            self.assertTrue(result["hardware_ready"])

    def test_generate_reports_marks_simulation_ready_when_hardware_gate_missing(self) -> None:
        with self._workspace_tmpdir() as root:
            inputs = self._make_inputs(root)
            out_dir = root / "release_gate"
            result = generate_reports(
                run_id="test",
                out_dir=out_dir,
                unittest_log=inputs["unittest_log"],
                w7_summary=inputs["w7"],
                w13_summary=inputs["w13"],
                operations_report=inputs["operations"],
                compliance_manifest=inputs["compliance_manifest"],
                hardware_gate_report=None,
            )
            self.assertEqual(result["verdict"], "simulation_ready")
            self.assertTrue(result["simulation_ready"])
            self.assertFalse(result["hardware_ready"])


if __name__ == "__main__":
    unittest.main()
