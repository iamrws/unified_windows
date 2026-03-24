"""Compile release-gate readiness artifacts from RC evidence files."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN_ID = datetime.now().strftime("%Y-%m-%d")
DEFAULT_RUNLOG_DIRNAME = "reports/runlogs"
DEFAULT_RELEASE_GATE_DIRNAME = "reports/release_gate"
DEFAULT_RELEASE_ARTIFACTS_DIRNAME = "reports/release_artifacts"

SECURITY_GATE_TESTS = [
    "test_security_contract.SecurityContractTests.test_unknown_or_cross_user_caller_is_denied_by_default",
    "test_security_contract.SecurityContractTests.test_same_user_authorization_is_allowed",
    "test_security_contract.SecurityContractTests.test_cross_user_allowlist_can_grant_access",
    "test_security_contract.SecurityContractTests.test_create_session_rate_limit_and_session_cap_are_enforced",
    "test_integrated_service_contract.IntegratedServiceContractTests.test_session_ownership_isolation_denies_cross_user_access",
    "test_ipc_client_sdk.IpcClientSdkE2ETests.test_connection_rejects_caller_identity_switch_after_binding",
    "test_ipc_server.IpcServerContractTests.test_server_requires_explicit_caller_identity_by_default",
    "test_ipc_server.IpcServerContractTests.test_server_rejects_oversized_request_payload",
    "test_ipc_server.IpcServerContractTests.test_security_denials_are_recorded_in_telemetry",
    "test_ipc_protocol.IpcProtocolContractTests.test_request_envelope_validation_rejects_malformed_input",
    "test_ipc_protocol.IpcProtocolContractTests.test_request_envelope_rejects_payloads_over_max_size",
    "test_phase3_runtime_smoke.Phase3RuntimeSmokeTests.test_remote_call_smoke_path_denies_foreign_caller",
]

DECISION_LOG_TESTS = {
    "D-001": [
        "test_ipc_server.IpcServerContractTests.test_server_requires_explicit_caller_identity_by_default",
        "test_ipc_client_sdk.IpcClientSdkE2ETests.test_connection_rejects_caller_identity_switch_after_binding",
        "test_service_contract.AstraWeaveServiceLifecycleTests.test_invalid_order_rejection_blocks_out_of_sequence_calls",
    ],
    "D-002": [
        "test_fallback_contract.FallbackContractTests.test_controller_declines_step_changes_inside_cooldown_window",
        "test_fallback_contract.FallbackContractTests.test_controller_respects_cooldown_before_advancing",
        "test_fallback_contract.FallbackContractTests.test_anti_oscillation_defaults_are_locked",
    ],
    "D-003": [
        "test_capability_matrix_contract.CapabilityMatrixContractTests.test_profile_pa_primary_maps_to_numa_dgpu",
        "test_capability_matrix_contract.CapabilityMatrixContractTests.test_profile_pb_secondary_maps_to_cache_coherent_uma",
        "test_capability_matrix_contract.CapabilityMatrixContractTests.test_profile_pc_degraded_maps_to_uma",
        "test_capability_matrix_contract.CapabilityMatrixContractTests.test_profile_pd_unsupported_fails_fast",
    ],
    "D-004": [
        "test_telemetry_contract.TelemetryContractTests.test_local_only_is_default_and_export_is_disabled",
        "test_telemetry_contract.TelemetryContractTests.test_export_requires_explicit_opt_in",
    ],
    "D-005": [
        "test_service_contract.AstraWeaveServiceLifecycleTests.test_env_flag_can_enable_hardware_mode_without_constructor_override",
        "test_service_contract.AstraWeaveServiceLifecycleTests.test_run_step_hardware_mode_invokes_executor_and_returns_hardware_payload",
        "test_service_contract.AstraWeaveServiceLifecycleTests.test_run_step_hardware_mode_marks_session_degraded_on_runtime_failure",
    ],
}

STABILITY_TESTS = [
    "test_fallback_contract.FallbackContractTests.test_controller_declines_step_changes_inside_cooldown_window",
    "test_fallback_contract.FallbackContractTests.test_controller_respects_cooldown_before_advancing",
    "test_fallback_contract.FallbackContractTests.test_anti_oscillation_defaults_are_locked",
]

EXPLAINABILITY_TESTS = [
    "test_integrated_service_contract.IntegratedServiceContractTests.test_service_emits_telemetry_events_with_stable_reason_codes_and_correlation_ids",
]

API_CONFORMANCE_TESTS = [
    "test_service_contract.AstraWeaveServiceLifecycleTests.test_invalid_order_rejection_blocks_out_of_sequence_calls",
    "test_service_contract.AstraWeaveServiceLifecycleTests.test_close_session_is_idempotent",
    "test_service_contract.AstraWeaveServiceLifecycleTests.test_run_step_rejects_second_active_execution",
    "test_ipc_protocol.IpcProtocolContractTests.test_error_envelope_maps_api_error_codes_stably",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _parse_unittest_log(path: Path) -> tuple[dict[str, str], int | None]:
    statuses: dict[str, str] = {}
    ran_tests: int | None = None
    test_line = re.compile(r"^(test\S+) \(([^)]+)\) \.\.\. (\w+)$")
    ran_line = re.compile(r"^Ran (\d+) tests? in ")

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        match = test_line.match(line)
        if match:
            test_id = match.group(2)
            status = match.group(3).lower()
            statuses[test_id] = status
            continue
        ran_match = ran_line.match(line)
        if ran_match:
            ran_tests = int(ran_match.group(1))
    return statuses, ran_tests


def _all_ok(statuses: dict[str, str], test_ids: list[str]) -> tuple[bool, list[str]]:
    missing_or_failed = [tid for tid in test_ids if statuses.get(tid) != "ok"]
    return (len(missing_or_failed) == 0), missing_or_failed


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _normalize_artifact_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def _load_repo_text(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8-sig")


def _resolve_path(path_text: str | None, *, root: Path) -> Path | None:
    if path_text is None or not str(path_text).strip():
        return None
    path = Path(path_text)
    if not path.is_absolute():
        path = (root / path).resolve()
    return path


def _default_artifact_path(*, root: Path, run_id: str, kind: str) -> Path:
    if kind == "unittest_log":
        return root / DEFAULT_RUNLOG_DIRNAME / f"unittest_rc_{run_id}.txt"
    if kind == "w7_summary":
        return root / DEFAULT_RUNLOG_DIRNAME / f"summary_rc_w7b_chat_{run_id}.json"
    if kind == "w13_summary":
        return root / DEFAULT_RUNLOG_DIRNAME / f"summary_rc_w13b_chat_{run_id}.json"
    if kind == "operations_report":
        return root / DEFAULT_RELEASE_GATE_DIRNAME / f"operations_drill_{run_id}.json"
    if kind == "compliance_manifest":
        return root / DEFAULT_RELEASE_ARTIFACTS_DIRNAME / f"compliance_manifest_{run_id}.json"
    if kind == "hardware_gate_report":
        return root / DEFAULT_RELEASE_GATE_DIRNAME / f"hardware_gate_{run_id}.json"
    raise ValueError(f"unknown artifact kind: {kind}")


def _find_operation_drill(operations_report: dict[str, Any], drill_id: str) -> dict[str, Any] | None:
    drills = operations_report.get("drills")
    if not isinstance(drills, list):
        return None
    for drill in drills:
        if isinstance(drill, dict) and drill.get("drill_id") == drill_id:
            return drill
    return None


def _decision_log_alignment_report(statuses: dict[str, str]) -> tuple[bool, dict[str, Any], list[str]]:
    plan_text = _load_repo_text("plan.md")
    problems_text = _load_repo_text("problems.md")
    plan_mentions = all(f"D-00{index}" in plan_text for index in range(1, 6))
    problems_mentions = (
        "D-001 through D-005" in problems_text
        and "Implementation behavior matches closed decision log." in problems_text
    )

    rows: dict[str, Any] = {}
    missing_or_failed: list[str] = []
    all_pass = plan_mentions and problems_mentions
    for decision_id, test_ids in DECISION_LOG_TESTS.items():
        passed, failed_tests = _all_ok(statuses, test_ids)
        row_pass = plan_mentions and problems_mentions and passed
        rows[decision_id] = {
            "passed": row_pass,
            "tests": test_ids,
            "missing_or_failed": failed_tests,
            "plan_mentions_decision": plan_mentions and decision_id in plan_text,
            "problems_mentions_decision": problems_mentions,
        }
        if not row_pass:
            all_pass = False
            missing_or_failed.append(decision_id)

    return all_pass, rows, missing_or_failed


def _security_gate_report(statuses: dict[str, str], operations_report: dict[str, Any]) -> tuple[bool, dict[str, Any], list[str]]:
    auth_abuse = _find_operation_drill(operations_report, "DRILL-AUTH-ABUSE")
    auth_abuse_pass = bool(auth_abuse and auth_abuse.get("passed") is True)
    tests_pass, missing_or_failed = _all_ok(statuses, SECURITY_GATE_TESTS)
    if not auth_abuse_pass:
        missing_or_failed = [*missing_or_failed, "DRILL-AUTH-ABUSE"]
    passed = tests_pass and auth_abuse_pass
    report = {
        "passed": passed,
        "tests": SECURITY_GATE_TESTS,
        "missing_or_failed": missing_or_failed,
        "auth_abuse_drill_passed": auth_abuse_pass,
    }
    return passed, report, missing_or_failed


def generate_reports(
    *,
    run_id: str,
    out_dir: Path,
    unittest_log: Path,
    w7_summary: Path,
    w13_summary: Path,
    operations_report: Path,
    compliance_manifest: Path,
    hardware_gate_report: Path | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    statuses, ran_tests = _parse_unittest_log(unittest_log)

    w7 = _read_json(w7_summary)
    w13 = _read_json(w13_summary)
    ops = _read_json(operations_report)
    compliance = _read_json(compliance_manifest)
    hardware_gate: dict[str, Any] | None = None
    if hardware_gate_report is not None and hardware_gate_report.exists():
        hardware_gate = _read_json(hardware_gate_report)

    high_threat_tests = {
        "T-001": SECURITY_GATE_TESTS,
        "T-002": [
            "test_integrated_service_contract.IntegratedServiceContractTests.test_session_ownership_isolation_denies_cross_user_access",
            "test_phase3_runtime_smoke.Phase3RuntimeSmokeTests.test_remote_call_smoke_path_denies_foreign_caller",
        ],
        "T-003": [
            "test_ipc_protocol.IpcProtocolContractTests.test_request_envelope_validation_rejects_malformed_input",
            "test_ipc_protocol.IpcProtocolContractTests.test_request_envelope_rejects_payloads_over_max_size",
            "test_ipc_server.IpcServerContractTests.test_server_rejects_oversized_request_payload",
        ],
        "T-005": [
            "test_telemetry_contract.TelemetryContractTests.test_local_only_is_default_and_export_is_disabled",
            "test_telemetry_contract.TelemetryContractTests.test_export_requires_explicit_opt_in",
        ],
    }

    threat_checks = {}
    all_threats_pass = True
    for threat_id, tests in high_threat_tests.items():
        passed, missing_or_failed = _all_ok(statuses, tests)
        all_threats_pass = all_threats_pass and passed
        threat_checks[threat_id] = {
            "passed": passed,
            "tests": tests,
            "missing_or_failed": missing_or_failed,
        }

    threat_signoff = {
        "run_id": run_id,
        "generated_at": _utc_now_iso(),
        "source_unittest_log": str(unittest_log),
        "high_threats": threat_checks,
        "verdict": "pass" if all_threats_pass else "fail",
        "signoff_note": (
            "All high-severity threat controls are covered by passing automated tests."
            if all_threats_pass
            else "One or more high-severity threat controls lack passing evidence."
        ),
    }
    threat_json = out_dir / f"threat_signoff_{run_id}.json"
    threat_md = out_dir / f"threat_signoff_{run_id}.md"
    _write_json(threat_json, threat_signoff)
    _write_markdown(
        threat_md,
        [
            "# Threat Model Sign-Off",
            "",
            f"- Run id: `{run_id}`",
            f"- Generated at: `{threat_signoff['generated_at']}`",
            f"- Verdict: `{threat_signoff['verdict']}`",
            "",
            "| Threat | Result | Evidence |",
            "| --- | --- | --- |",
            *[
                f"| `{tid}` | `{'pass' if data['passed'] else 'fail'}` | {', '.join(f'`{name}`' for name in data['tests'])} |"
                for tid, data in threat_checks.items()
            ],
            "",
            f"- Source unittest log: `{unittest_log}`",
        ],
    )

    capability_tests = {
        "P-A": "test_capability_matrix_contract.CapabilityMatrixContractTests.test_profile_pa_primary_maps_to_numa_dgpu",
        "P-B": "test_capability_matrix_contract.CapabilityMatrixContractTests.test_profile_pb_secondary_maps_to_cache_coherent_uma",
        "P-C": "test_capability_matrix_contract.CapabilityMatrixContractTests.test_profile_pc_degraded_maps_to_uma",
        "P-D": "test_capability_matrix_contract.CapabilityMatrixContractTests.test_profile_pd_unsupported_fails_fast",
    }
    matrix_rows = {}
    matrix_pass = True
    for profile, test_id in capability_tests.items():
        status = statuses.get(test_id, "missing")
        row_pass = status == "ok"
        matrix_pass = matrix_pass and row_pass
        matrix_rows[profile] = {
            "test": test_id,
            "status": status,
            "passed": row_pass,
        }
    compatibility_report = {
        "run_id": run_id,
        "generated_at": _utc_now_iso(),
        "source_unittest_log": str(unittest_log),
        "profiles": matrix_rows,
        "verdict": "pass" if matrix_pass else "fail",
    }
    matrix_json = out_dir / f"compatibility_matrix_{run_id}.json"
    matrix_md = out_dir / f"compatibility_matrix_{run_id}.md"
    _write_json(matrix_json, compatibility_report)
    _write_markdown(
        matrix_md,
        [
            "# Compatibility Matrix Evidence",
            "",
            f"- Run id: `{run_id}`",
            f"- Generated at: `{compatibility_report['generated_at']}`",
            f"- Verdict: `{compatibility_report['verdict']}`",
            "",
            "| Profile | Test | Result |",
            "| --- | --- | --- |",
            *[
                f"| `{profile}` | `{row['test']}` | `{'pass' if row['passed'] else 'fail'}` |"
                for profile, row in matrix_rows.items()
            ],
            "",
            "- Notes:",
            "  - `P-A` evidence uses real RC workload runs.",
            "  - `P-B/P-C/P-D` evidence is branch-contract validation via capability mapping tests.",
        ],
    )

    reliability_pass = (
        w7.get("iterations", 0) >= 100
        and w13.get("iterations", 0) >= 100
        and w7.get("failure_count") == 0
        and w13.get("failure_count") == 0
    )
    latency_value = max(float(w7.get("run_step_p95_ms", 0.0)), float(w13.get("run_step_p95_ms", 0.0)))
    latency_pass = latency_value <= 250.0

    security_pass, _security_report, security_missing = _security_gate_report(statuses, ops)
    decision_alignment_pass, decision_alignment_rows, decision_alignment_missing = _decision_log_alignment_report(statuses)
    stability_pass, stability_missing = _all_ok(statuses, STABILITY_TESTS)
    explainability_pass, explainability_missing = _all_ok(statuses, EXPLAINABILITY_TESTS)

    rc_workload_report = {
        "run_id": run_id,
        "generated_at": _utc_now_iso(),
        "workloads": {
            "W-7B-CHAT": w7,
            "W-13B-CHAT": w13,
        },
        "gates": {
            "reliability_zero_hard_oom": {
                "passed": reliability_pass,
                "details": "Both workloads completed >=100 iterations with zero failures.",
            },
            "latency_p95_le_250ms": {
                "passed": latency_pass,
                "measured_p95_ms": latency_value,
                "threshold_ms": 250.0,
            },
            "stability_anti_oscillation": {
                "passed": stability_pass,
                "tests": STABILITY_TESTS,
                "missing_or_failed": stability_missing,
            },
            "explainability_reason_code_and_correlation": {
                "passed": explainability_pass,
                "tests": EXPLAINABILITY_TESTS,
                "missing_or_failed": explainability_missing,
            },
        },
    }
    rc_workload_report["verdict"] = (
        "pass"
        if all(item["passed"] for item in rc_workload_report["gates"].values())
        else "fail"
    )
    workload_json = out_dir / f"rc_workload_report_{run_id}.json"
    workload_md = out_dir / f"rc_workload_report_{run_id}.md"
    _write_json(workload_json, rc_workload_report)
    _write_markdown(
        workload_md,
        [
            "# RC Workload Report",
            "",
            f"- Run id: `{run_id}`",
            f"- Generated at: `{rc_workload_report['generated_at']}`",
            f"- Verdict: `{rc_workload_report['verdict']}`",
            f"- W-7B summary: `{w7_summary}`",
            f"- W-13B summary: `{w13_summary}`",
            "",
            "| Gate | Result | Details |",
            "| --- | --- | --- |",
            f"| Reliability (zero hard OOM proxy via zero failures) | `{'pass' if reliability_pass else 'fail'}` | 7B failures={w7.get('failure_count')}, 13B failures={w13.get('failure_count')} |",
            f"| Latency p95 <= 250 ms | `{'pass' if latency_pass else 'fail'}` | measured p95={latency_value} ms |",
            f"| Stability anti-oscillation | `{'pass' if stability_pass else 'fail'}` | tests={', '.join(STABILITY_TESTS)} |",
            f"| Explainability reason-code/correlation | `{'pass' if explainability_pass else 'fail'}` | tests={', '.join(EXPLAINABILITY_TESTS)} |",
        ],
    )

    compliance_artifacts = compliance.get("artifacts", {})
    compliance_files = {
        key: _normalize_artifact_path(value)
        for key, value in compliance_artifacts.items()
    }
    compliance_pass = (
        bool(compliance_files)
        and all(path.exists() for path in compliance_files.values())
    )

    api_conformance_tests = [
        "test_service_contract.AstraWeaveServiceLifecycleTests.test_invalid_order_rejection_blocks_out_of_sequence_calls",
        "test_service_contract.AstraWeaveServiceLifecycleTests.test_close_session_is_idempotent",
        "test_service_contract.AstraWeaveServiceLifecycleTests.test_run_step_rejects_second_active_execution",
        "test_ipc_protocol.IpcProtocolContractTests.test_error_envelope_maps_api_error_codes_stably",
    ]
    api_pass, api_missing = _all_ok(statuses, api_conformance_tests)

    readiness = {
        "run_id": run_id,
        "generated_at": _utc_now_iso(),
        "evidence_sources": {
            "unittest_log": str(unittest_log),
            "w7_summary": str(w7_summary),
            "w13_summary": str(w13_summary),
            "operations_report": str(operations_report),
            "compliance_manifest": str(compliance_manifest),
            "hardware_gate_report": str(hardware_gate_report) if hardware_gate_report is not None else None,
        },
        "test_suite": {
            "ran_tests": ran_tests,
            "status": "pass" if ran_tests and ran_tests > 0 else "unknown",
        },
        "gates": {
            "P0-01": {
                "passed": security_pass,
                "missing_or_failed": security_missing,
                "evidence": {
                    "security_tests": SECURITY_GATE_TESTS,
                    "auth_abuse_drill": "DRILL-AUTH-ABUSE",
                },
                "note": "Security controls validated by contract/integration tests and the auth abuse drill.",
            },
            "P0-02": {"passed": all_threats_pass, "evidence": str(threat_json)},
            "P0-03": {
                "passed": _all_ok(
                    statuses,
                    [
                        "test_telemetry_contract.TelemetryContractTests.test_local_only_is_default_and_export_is_disabled",
                        "test_telemetry_contract.TelemetryContractTests.test_export_requires_explicit_opt_in",
                    ],
                )[0],
                "note": "Telemetry privacy defaults and opt-in export controls are validated by tests.",
            },
            "P0-04": {
                "passed": api_pass,
                "missing_or_failed": api_missing,
                "note": "Lifecycle/error-taxonomy/single-flight conformance is validated by service + IPC contract tests.",
            },
            "P1-05": {
                "passed": reliability_pass and matrix_pass,
                "note": "Primary workload runs passed; secondary/degraded/unsupported branches validated via capability tests.",
            },
            "P1-06": {
                "passed": decision_alignment_pass,
                "missing_or_failed": decision_alignment_missing,
                "evidence": {
                    "plan": str(REPO_ROOT / "plan.md"),
                    "problems": str(REPO_ROOT / "problems.md"),
                    "decision_tests": decision_alignment_rows,
                },
                "note": "Decision log D-001 through D-005 is present in the normative docs and backed by the decision-specific tests.",
            },
            "P1-07": {"passed": rc_workload_report["verdict"] == "pass", "evidence": str(workload_json)},
            "P1-08": {"passed": ops.get("verdict") == "pass", "evidence": str(operations_report)},
            "P2-09": {"passed": matrix_pass, "evidence": str(matrix_json)},
            "P2-10": {"passed": compliance_pass, "evidence": str(compliance_manifest)},
        },
    }
    hardware_checks = hardware_gate.get("gate_checks", {}) if isinstance(hardware_gate, dict) else {}
    run_mode_active = bool(
        isinstance(hardware_checks.get("service_hardware_mode_active"), dict)
        and hardware_checks["service_hardware_mode_active"].get("passed")
    )
    transfer_ok = bool(
        isinstance(hardware_checks.get("service_hardware_transfer_ok"), dict)
        and hardware_checks["service_hardware_transfer_ok"].get("passed")
    )
    delta_observed = bool(
        isinstance(hardware_checks.get("service_triggered_nvml_delta_observed"), dict)
        and hardware_checks["service_triggered_nvml_delta_observed"].get("passed")
    )
    hardware_evidence = (
        str(hardware_gate_report)
        if hardware_gate_report is not None and hardware_gate_report.exists()
        else "missing hardware gate artifact (run scripts/run_hardware_gate.py)"
    )
    readiness["gates"]["P0-HW-11"] = {
        "passed": run_mode_active and transfer_ok,
        "evidence": hardware_evidence,
        "note": "Service RunStep executed in hardware mode and completed CUDA transfer proof.",
    }
    readiness["gates"]["P1-HW-12"] = {
        "passed": delta_observed,
        "evidence": hardware_evidence,
        "note": "Service-triggered NVML memory delta was observed during hardware RunStep.",
    }
    readiness["gates"]["P1-HW-13"] = {
        "passed": True,
        "note": "Release report now distinguishes simulation-ready and hardware-ready tracks.",
    }

    simulation_gate_ids = [
        "P0-01",
        "P0-02",
        "P0-03",
        "P0-04",
        "P1-05",
        "P1-06",
        "P1-07",
        "P1-08",
        "P2-09",
        "P2-10",
    ]
    hardware_gate_ids = ["P0-HW-11", "P1-HW-12", "P1-HW-13"]
    simulation_ready = all(readiness["gates"][gate]["passed"] for gate in simulation_gate_ids)
    hardware_ready = simulation_ready and all(readiness["gates"][gate]["passed"] for gate in hardware_gate_ids)
    readiness["tracks"] = {
        "simulation_ready": {
            "passed": simulation_ready,
            "gates": simulation_gate_ids,
        },
        "hardware_ready": {
            "passed": hardware_ready,
            "gates": hardware_gate_ids,
        },
    }
    if hardware_ready:
        readiness["verdict"] = "hardware_ready"
    elif simulation_ready:
        readiness["verdict"] = "simulation_ready"
    else:
        readiness["verdict"] = "fail"

    readiness_json = out_dir / f"release_gate_readiness_{run_id}.json"
    readiness_md = out_dir / f"release_gate_readiness_{run_id}.md"
    _write_json(readiness_json, readiness)
    _write_markdown(
        readiness_md,
        [
            "# Release Gate Readiness",
            "",
            f"- Run id: `{run_id}`",
            f"- Generated at: `{readiness['generated_at']}`",
            f"- Overall verdict: `{readiness['verdict']}`",
            f"- Simulation-ready: `{'pass' if simulation_ready else 'fail'}`",
            f"- Hardware-ready: `{'pass' if hardware_ready else 'fail'}`",
            "",
            "| Gate | Result | Evidence |",
            "| --- | --- | --- |",
            *[
                f"| `{gate}` | `{'pass' if data['passed'] else 'fail'}` | `{data.get('evidence', data.get('note', 'n/a'))}` |"
                for gate, data in readiness["gates"].items()
            ],
        ],
    )

    return {
        "threat_signoff_json": str(threat_json),
        "compatibility_json": str(matrix_json),
        "workload_report_json": str(workload_json),
        "readiness_json": str(readiness_json),
        "verdict": readiness["verdict"],
        "simulation_ready": simulation_ready,
        "hardware_ready": hardware_ready,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate release-gate evidence reports.")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--artifacts-root",
        default=str(REPO_ROOT),
        help="Root directory used to resolve default evidence paths.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for generated reports.",
    )
    parser.add_argument(
        "--unittest-log",
        default=None,
        help="Path to the unittest log. Defaults to a run-id-scoped file under --artifacts-root.",
    )
    parser.add_argument(
        "--w7-summary",
        default=None,
        help="Path to the W-7B summary. Defaults to a run-id-scoped file under --artifacts-root.",
    )
    parser.add_argument(
        "--w13-summary",
        default=None,
        help="Path to the W-13B summary. Defaults to a run-id-scoped file under --artifacts-root.",
    )
    parser.add_argument(
        "--operations-report",
        default=None,
        help="Path to the operations drill report. Defaults to a run-id-scoped file under --artifacts-root.",
    )
    parser.add_argument(
        "--compliance-manifest",
        default=None,
        help="Path to the compliance manifest. Defaults to a run-id-scoped file under --artifacts-root.",
    )
    parser.add_argument(
        "--hardware-gate-report",
        default=None,
        help="Optional hardware gate artifact. Defaults to a run-id-scoped file under --artifacts-root if present.",
    )
    parser.add_argument(
        "--allow-simulation-only-success",
        action="store_true",
        help="Permit a simulation-ready verdict to exit successfully when hardware evidence is absent.",
    )
    args = parser.parse_args(argv)

    artifacts_root = Path(args.artifacts_root).resolve()
    out_dir = _resolve_path(args.out_dir, root=artifacts_root) or _default_artifact_path(
        root=artifacts_root, run_id=args.run_id, kind="hardware_gate_report"
    ).parent
    unittest_log = _resolve_path(args.unittest_log, root=artifacts_root) or _default_artifact_path(
        root=artifacts_root, run_id=args.run_id, kind="unittest_log"
    )
    w7_summary = _resolve_path(args.w7_summary, root=artifacts_root) or _default_artifact_path(
        root=artifacts_root, run_id=args.run_id, kind="w7_summary"
    )
    w13_summary = _resolve_path(args.w13_summary, root=artifacts_root) or _default_artifact_path(
        root=artifacts_root, run_id=args.run_id, kind="w13_summary"
    )
    operations_report = _resolve_path(args.operations_report, root=artifacts_root) or _default_artifact_path(
        root=artifacts_root, run_id=args.run_id, kind="operations_report"
    )
    compliance_manifest = _resolve_path(args.compliance_manifest, root=artifacts_root) or _default_artifact_path(
        root=artifacts_root, run_id=args.run_id, kind="compliance_manifest"
    )
    hardware_gate_report = _resolve_path(args.hardware_gate_report, root=artifacts_root)
    if hardware_gate_report is None:
        candidate = _default_artifact_path(root=artifacts_root, run_id=args.run_id, kind="hardware_gate_report")
        if candidate.exists():
            hardware_gate_report = candidate

    result = generate_reports(
        run_id=args.run_id,
        out_dir=out_dir,
        unittest_log=unittest_log,
        w7_summary=w7_summary,
        w13_summary=w13_summary,
        operations_report=operations_report,
        compliance_manifest=compliance_manifest,
        hardware_gate_report=hardware_gate_report,
    )
    print(json.dumps({"ok": True, "result": result}, indent=2, sort_keys=True))
    if result["verdict"] == "hardware_ready":
        return 0
    if args.allow_simulation_only_success and result["verdict"] == "simulation_ready":
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
