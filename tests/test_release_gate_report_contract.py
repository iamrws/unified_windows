"""Contract tests for release-gate readiness split reporting."""

from __future__ import annotations

import io
import json
import shutil
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import scripts.generate_release_gate_report as release_gate


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    _write(path, json.dumps(payload, sort_keys=True))


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

    @contextmanager
    def _patched_repo_root(self, root: Path):
        with patch.object(release_gate, "REPO_ROOT", root):
            yield

    def _all_test_ids(self) -> set[str]:
        ids: set[str] = set()
        ids.update(release_gate.SECURITY_GATE_TESTS)
        ids.update(release_gate.STABILITY_TESTS)
        ids.update(release_gate.EXPLAINABILITY_TESTS)
        ids.update(release_gate.API_CONFORMANCE_TESTS)
        for tests in release_gate.DECISION_LOG_TESTS.values():
            ids.update(tests)
        ids.update(
            {
                "test_integrated_service_contract.IntegratedServiceContractTests.test_session_ownership_isolation_denies_cross_user_access",
                "test_phase3_runtime_smoke.Phase3RuntimeSmokeTests.test_remote_call_smoke_path_denies_foreign_caller",
                "test_ipc_protocol.IpcProtocolContractTests.test_request_envelope_validation_rejects_malformed_input",
                "test_ipc_protocol.IpcProtocolContractTests.test_request_envelope_rejects_payloads_over_max_size",
                "test_ipc_server.IpcServerContractTests.test_server_rejects_oversized_request_payload",
                "test_telemetry_contract.TelemetryContractTests.test_local_only_is_default_and_export_is_disabled",
                "test_telemetry_contract.TelemetryContractTests.test_export_requires_explicit_opt_in",
            }
        )
        return ids

    def _make_unittest_log(self, root: Path, run_id: str, test_ids: set[str]) -> Path:
        lines = [
            f"test_case_{index:03d} ({test_id}) ... ok"
            for index, test_id in enumerate(sorted(test_ids), start=1)
        ]
        lines.append("")
        lines.append(f"Ran {len(test_ids)} tests in 0.123s")
        lines.append("")
        lines.append("OK")
        path = root / "reports" / "runlogs" / f"unittest_rc_{run_id}.txt"
        _write(path, "\n".join(lines))
        return path

    def _make_inputs(self, root: Path, *, run_id: str, include_hardware_gate: bool = True) -> dict[str, Path]:
        test_ids = self._all_test_ids()
        unittest_log = self._make_unittest_log(root, run_id, test_ids)
        runlog_dir = root / "reports" / "runlogs"
        gate_dir = root / "reports" / "release_gate"
        artifact_dir = root / "reports" / "release_artifacts"
        artifact_file = artifact_dir / "placeholder.json"
        compliance_manifest = artifact_dir / f"compliance_manifest_{run_id}.json"
        w7 = runlog_dir / f"summary_rc_w7b_chat_{run_id}.json"
        w13 = runlog_dir / f"summary_rc_w13b_chat_{run_id}.json"
        operations = gate_dir / f"operations_drill_{run_id}.json"
        hardware_gate = gate_dir / f"hardware_gate_{run_id}.json"

        _write_json(w7, {"iterations": 100, "failure_count": 0, "run_step_p95_ms": 181.0})
        _write_json(w13, {"iterations": 100, "failure_count": 0, "run_step_p95_ms": 183.0})
        _write_json(
            operations,
            {
                "verdict": "pass",
                "drills": [
                    {"drill_id": "DRILL-AUTH-ABUSE", "passed": True},
                    {"drill_id": "DRILL-BUDGET-CONTRACTION", "passed": True},
                    {"drill_id": "DRILL-ROLLBACK", "passed": True},
                ],
            },
        )
        _write_json(artifact_file, {"ok": True})
        _write_json(
            compliance_manifest,
            {"artifacts": {"placeholder": str(artifact_file)}},
        )
        if include_hardware_gate:
            _write_json(
                hardware_gate,
                {
                    "gate_checks": {
                        "service_hardware_mode_active": {"passed": True},
                        "service_hardware_transfer_ok": {"passed": True},
                        "service_triggered_nvml_delta_observed": {"passed": True},
                    }
                },
            )

        return {
            "unittest_log": unittest_log,
            "w7": w7,
            "w13": w13,
            "operations": operations,
            "compliance_manifest": compliance_manifest,
            "hardware_gate": hardware_gate,
        }

    def _make_repo_docs(self, root: Path, *, include_d005: bool = True, include_validation_phrase: bool = True) -> None:
        decision_rows = [
            "| D-001 | Runtime vs framework policy boundary | AstraWeave owns cross-tier placement, residency, migration, fallback, and observability. | Core Runtime Lead | 2026-03-24 |",
            "| D-002 | Tunability level | Stability profile is default with safe knobs only. Throughput profile is opt-in and cannot disable safety rails. | Product + Runtime | 2026-03-24 |",
            "| D-003 | Minimum hardware bar | v1 acceptance target is Windows 11 + 8 GB dGPU + >= 32 GB RAM; design-optimized target remains 128 GB RAM host. | Product | 2026-03-24 |",
            "| D-004 | Telemetry surface default | Local-only telemetry is default. Export and diagnostic bundle creation require explicit operator action. | Security + Product | 2026-03-24 |",
        ]
        if include_d005:
            decision_rows.append(
                "| D-005 | Hardware strategy contradiction | dGPU reliability-first is primary release path. CacheCoherentUMA is secondary fast path with separate acceptance gates. | Architecture Council | 2026-03-24 |"
            )
        _write(
            root / "plan.md",
            "\n".join(
                [
                    "# AstraWeave v1 Execution Plan",
                    "",
                    "| ID | Decision | Outcome | Owner | Date |",
                    "| --- | --- | --- | --- | --- |",
                    *decision_rows,
                ]
            ),
        )
        validation_line = (
            "- [x] P1-06 Implementation behavior matches closed decision log."
            if include_validation_phrase
            else "- [x] P1-06 Placeholder"
        )
        _write(
            root / "problems.md",
            "\n".join(
                [
                    "# Problems",
                    "",
                    "Decision status:",
                    "",
                    "- Gate evidence required: implementation docs and behavior align with decisions D-001 through D-005.",
                    validation_line,
                    "",
                    "## Release Readiness Checklist",
                    "",
                    "- [x] P0-01 Security controls validated by auth/isolation/abuse test results.",
                    "- [x] P1-06 Implementation behavior matches closed decision log.",
                ]
            ),
        )

    def test_generate_reports_marks_hardware_ready_when_full_evidence_passes(self) -> None:
        with self._workspace_tmpdir() as root:
            self._make_repo_docs(root)
            inputs = self._make_inputs(root, run_id="test", include_hardware_gate=True)
            out_dir = root / "reports" / "release_gate"
            with self._patched_repo_root(root):
                result = release_gate.generate_reports(
                    run_id="test",
                    out_dir=out_dir,
                    unittest_log=inputs["unittest_log"],
                    w7_summary=inputs["w7"],
                    w13_summary=inputs["w13"],
                    operations_report=inputs["operations"],
                    compliance_manifest=inputs["compliance_manifest"],
                    hardware_gate_report=inputs["hardware_gate"],
                )
            self.assertEqual(result["verdict"], "hardware_ready")
            self.assertTrue(result["simulation_ready"])
            self.assertTrue(result["hardware_ready"])

    def test_generate_reports_requires_security_drill_and_contract_coverage(self) -> None:
        with self._workspace_tmpdir() as root:
            self._make_repo_docs(root)
            inputs = self._make_inputs(root, run_id="test", include_hardware_gate=True)
            log_text = inputs["unittest_log"].read_text(encoding="utf-8")
            missing_test = release_gate.SECURITY_GATE_TESTS[0]
            filtered = "\n".join(
                line for line in log_text.splitlines() if missing_test not in line
            )
            inputs["unittest_log"].write_text(filtered, encoding="utf-8")
            with self._patched_repo_root(root):
                result = release_gate.generate_reports(
                    run_id="test",
                    out_dir=root / "reports" / "release_gate",
                    unittest_log=inputs["unittest_log"],
                    w7_summary=inputs["w7"],
                    w13_summary=inputs["w13"],
                    operations_report=inputs["operations"],
                    compliance_manifest=inputs["compliance_manifest"],
                    hardware_gate_report=inputs["hardware_gate"],
                )
            self.assertFalse(result["simulation_ready"])
            readiness = json.loads(
                (root / "reports" / "release_gate" / "release_gate_readiness_test.json").read_text(encoding="utf-8")
            )
            self.assertFalse(readiness["gates"]["P0-01"]["passed"])
            self.assertIn(missing_test, readiness["gates"]["P0-01"]["missing_or_failed"])

    def test_generate_reports_requires_decision_alignment_coverage(self) -> None:
        with self._workspace_tmpdir() as root:
            self._make_repo_docs(root, include_d005=False)
            inputs = self._make_inputs(root, run_id="test", include_hardware_gate=True)
            with self._patched_repo_root(root):
                result = release_gate.generate_reports(
                    run_id="test",
                    out_dir=root / "reports" / "release_gate",
                    unittest_log=inputs["unittest_log"],
                    w7_summary=inputs["w7"],
                    w13_summary=inputs["w13"],
                    operations_report=inputs["operations"],
                    compliance_manifest=inputs["compliance_manifest"],
                    hardware_gate_report=inputs["hardware_gate"],
                )
            self.assertFalse(result["simulation_ready"])
            readiness = json.loads(
                (root / "reports" / "release_gate" / "release_gate_readiness_test.json").read_text(encoding="utf-8")
            )
            self.assertFalse(readiness["gates"]["P1-06"]["passed"])
            self.assertIn("D-005", readiness["gates"]["P1-06"]["missing_or_failed"])

    def test_main_requires_hardware_verdict_by_default(self) -> None:
        with self._workspace_tmpdir() as root:
            self._make_repo_docs(root)
            self._make_inputs(root, run_id="strict", include_hardware_gate=True)
            stdout = io.StringIO()
            with self._patched_repo_root(root), redirect_stdout(stdout):
                exit_code = release_gate.main(
                    [
                        "--artifacts-root",
                        str(root),
                        "--run-id",
                        "strict",
                        "--out-dir",
                        "reports/release_gate",
                    ]
                )
            self.assertEqual(exit_code, 0)
            self.assertIn('"verdict": "hardware_ready"', stdout.getvalue())

    def test_main_allows_simulation_only_success_when_explicitly_enabled(self) -> None:
        with self._workspace_tmpdir() as root:
            self._make_repo_docs(root)
            self._make_inputs(root, run_id="sim", include_hardware_gate=False)
            stdout = io.StringIO()
            with self._patched_repo_root(root), redirect_stdout(stdout):
                exit_code = release_gate.main(
                    [
                        "--artifacts-root",
                        str(root),
                        "--run-id",
                        "sim",
                        "--out-dir",
                        "reports/release_gate",
                        "--allow-simulation-only-success",
                    ]
                )
            self.assertEqual(exit_code, 0)
            self.assertIn('"verdict": "simulation_ready"', stdout.getvalue())

    def test_main_does_not_fall_back_to_legacy_dated_artifacts(self) -> None:
        with self._workspace_tmpdir() as root:
            self._make_repo_docs(root)
            self._make_inputs(root, run_id="2099-01-01", include_hardware_gate=False)
            legacy_gate = root / "reports" / "release_gate" / "hardware_gate_2026-03-24.json"
            _write_json(
                legacy_gate,
                {
                    "gate_checks": {
                        "service_hardware_mode_active": {"passed": True},
                        "service_hardware_transfer_ok": {"passed": True},
                        "service_triggered_nvml_delta_observed": {"passed": True},
                    }
                },
            )
            stdout = io.StringIO()
            with self._patched_repo_root(root), redirect_stdout(stdout):
                exit_code = release_gate.main(
                    [
                        "--artifacts-root",
                        str(root),
                        "--run-id",
                        "2099-01-01",
                        "--out-dir",
                        "reports/release_gate",
                    ]
                )
            self.assertEqual(exit_code, 1)
            self.assertIn('"verdict": "simulation_ready"', stdout.getvalue())
            self.assertNotIn('"verdict": "hardware_ready"', stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
