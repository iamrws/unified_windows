"""Run required AstraWeave v1 operations drills and emit evidence artifacts.

This script executes the three mandatory drills from docs/operations-readiness.md:
1) auth abuse drill
2) budget contraction and recovery drill
3) rollback drill
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from astrawave.errors import ApiError, ApiErrorCode
from astrawave.ipc_client import AstraWeaveIpcClient
from astrawave.security import CallerIdentity
from astrawave.service import DEFAULT_SERVICE_OWNER_SID, AstraWeaveService
from astrawave.service_host import AstraWeaveServiceHost, ServiceHostConfig
from astrawave.types import MemoryTier


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class DrillResult:
    drill_id: str
    title: str
    passed: bool
    expected: str
    observed: str
    owner: str
    started_at: str
    ended_at: str
    details: dict[str, Any]


def _run_auth_abuse_drill() -> DrillResult:
    started = _utc_now_iso()
    owner = CallerIdentity(user_sid=DEFAULT_SERVICE_OWNER_SID, pid=1101)
    foreign = CallerIdentity(user_sid="S-1-5-21-ops-foreign", pid=2201)
    service = AstraWeaveService()

    session_id = service.CreateSession(caller_identity=owner)
    service.LoadModel(session_id, "ops-drill-model", caller_identity=owner)

    passed = False
    observed = "Expected AW_ERR_AUTH_DENIED but no denial was raised"
    details: dict[str, Any] = {"session_id": session_id}
    try:
        service.GetPressure(session_id, caller_identity=foreign)
    except ApiError as exc:
        details["error_code"] = exc.code.value
        details["error_message"] = str(exc)
        if exc.code is ApiErrorCode.AUTH_DENIED:
            passed = True
            observed = "Foreign caller denied with AW_ERR_AUTH_DENIED as expected"
        else:
            observed = f"Unexpected denial code: {exc.code.value}"

    ended = _utc_now_iso()
    return DrillResult(
        drill_id="DRILL-AUTH-ABUSE",
        title="Unauthorized caller rejection",
        passed=passed,
        expected="Foreign caller is denied when attempting session access",
        observed=observed,
        owner="Security owner",
        started_at=started,
        ended_at=ended,
        details=details,
    )


def _run_budget_contraction_drill() -> DrillResult:
    started = _utc_now_iso()
    owner = CallerIdentity(user_sid=DEFAULT_SERVICE_OWNER_SID, pid=1102)
    service = AstraWeaveService()

    session_id = service.CreateSession(caller_identity=owner)
    service.LoadModel(session_id, "ops-drill-model", caller_identity=owner)
    service.RegisterTensor(
        session_id,
        "kv",
        64 * 1024 * 1024,
        caller_identity=owner,
    )
    service.SetTierHint(session_id, "kv", MemoryTier.HOT, caller_identity=owner)
    service.PrefetchPlan(session_id, caller_identity=owner)

    pressure_before = service.GetPressure(session_id, caller_identity=owner).pressure_level

    with service._lock:  # intentional internal mutation for synthetic budget drill
        session = service._sessions[session_id]
    with session.lock:
        original_budget = int(session.vram_budget_bytes)
        session.vram_budget_bytes = max(1, int(session.vram_used_bytes // 4))
        contracted_budget = int(session.vram_budget_bytes)

    pressure_contracted = service.GetPressure(session_id, caller_identity=owner).pressure_level
    run_result = service.RunStep(session_id, step_name="decode", caller_identity=owner)

    with session.lock:
        session.vram_budget_bytes = original_budget
    pressure_recovered = service.GetPressure(session_id, caller_identity=owner).pressure_level

    fallback_result = run_result.get("fallback_result")
    fallback_reason = (
        fallback_result.get("reason_code")
        if isinstance(fallback_result, dict)
        else None
    )
    correlation_id = run_result.get("correlation_id")

    passed = (
        pressure_contracted > pressure_before
        and pressure_contracted >= 0.75
        and pressure_recovered < pressure_contracted
        and isinstance(correlation_id, str)
        and bool(correlation_id)
        and isinstance(fallback_reason, str)
        and bool(fallback_reason)
    )

    if passed:
        observed = (
            "Budget contraction elevated pressure and triggered reason-coded fallback; "
            "pressure dropped after restoring budget"
        )
    else:
        observed = (
            "Budget drill did not satisfy expected pressure/fallback/correlation checks"
        )

    details = {
        "session_id": session_id,
        "pressure_before": pressure_before,
        "pressure_contracted": pressure_contracted,
        "pressure_recovered": pressure_recovered,
        "original_budget_bytes": original_budget,
        "contracted_budget_bytes": contracted_budget,
        "fallback_reason_code": fallback_reason,
        "correlation_id": correlation_id,
    }

    ended = _utc_now_iso()
    return DrillResult(
        drill_id="DRILL-BUDGET-CONTRACTION",
        title="Budget contraction and recovery",
        passed=passed,
        expected=(
            "Synthetic budget contraction raises pressure and emits reason-coded fallback; "
            "restored budget lowers pressure without hard failure"
        ),
        observed=observed,
        owner="Platform owner",
        started_at=started,
        ended_at=ended,
        details=details,
    )


def _format_tcp_endpoint(host_status_endpoint: Any) -> str:
    if isinstance(host_status_endpoint, tuple) and len(host_status_endpoint) >= 2:
        return f"tcp://{host_status_endpoint[0]}:{host_status_endpoint[1]}"
    raise RuntimeError(f"Unexpected host endpoint shape: {host_status_endpoint!r}")


def _run_rollback_drill() -> DrillResult:
    started = _utc_now_iso()
    owner = CallerIdentity(user_sid=DEFAULT_SERVICE_OWNER_SID, pid=1103)

    candidate_host = AstraWeaveServiceHost(
        config=ServiceHostConfig(endpoint="tcp://127.0.0.1:0")
    )
    previous_host = AstraWeaveServiceHost(
        config=ServiceHostConfig(endpoint="tcp://127.0.0.1:0")
    )

    candidate_status = candidate_host.start()
    candidate_endpoint = _format_tcp_endpoint(candidate_status.endpoint)
    with AstraWeaveIpcClient(endpoint=candidate_endpoint, default_caller=owner) as client:
        candidate_session = client.CreateSession()
        client.LoadModel(candidate_session, "ops-drill-model")
        client.CloseSession(candidate_session)
    stopped_candidate = candidate_host.stop()

    previous_status = previous_host.start()
    previous_endpoint = _format_tcp_endpoint(previous_status.endpoint)
    with AstraWeaveIpcClient(endpoint=previous_endpoint, default_caller=owner) as client:
        rollback_session = client.CreateSession()
        client.LoadModel(rollback_session, "ops-drill-model")
        client.CloseSession(rollback_session)
    stopped_previous = previous_host.stop()

    passed = (
        bool(candidate_endpoint)
        and bool(previous_endpoint)
        and stopped_candidate.running is False
        and stopped_previous.running is False
        and stopped_candidate.served_requests >= 1
        and stopped_previous.served_requests >= 1
    )

    observed = (
        "Candidate runtime served traffic, was stopped, and replacement runtime served traffic"
        if passed
        else "Rollback runtime drill did not complete expected start/stop/probe flow"
    )

    details = {
        "candidate_endpoint": candidate_endpoint,
        "candidate_served_requests": stopped_candidate.served_requests,
        "rollback_endpoint": previous_endpoint,
        "rollback_served_requests": stopped_previous.served_requests,
    }

    ended = _utc_now_iso()
    return DrillResult(
        drill_id="DRILL-ROLLBACK",
        title="Candidate-to-previous rollback validation",
        passed=passed,
        expected="Rollback path serves successful probe traffic after candidate shutdown",
        observed=observed,
        owner="Release owner",
        started_at=started,
        ended_at=ended,
        details=details,
    )


def _write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Operations Drill Report")
    lines.append("")
    lines.append(f"- Run id: `{report['run_id']}`")
    lines.append(f"- Started: `{report['started_at']}`")
    lines.append(f"- Ended: `{report['ended_at']}`")
    lines.append(f"- Verdict: `{report['verdict']}`")
    lines.append("")
    lines.append("| Drill | Owner | Result | Expected | Observed |")
    lines.append("| --- | --- | --- | --- | --- |")
    for drill in report["drills"]:
        result = "pass" if drill["passed"] else "fail"
        lines.append(
            f"| `{drill['drill_id']}` | {drill['owner']} | `{result}` | {drill['expected']} | {drill['observed']} |"
        )
    lines.append("")
    lines.append("## Drill Details")
    lines.append("")
    for drill in report["drills"]:
        lines.append(f"### {drill['drill_id']} - {drill['title']}")
        lines.append("")
        lines.append(f"- Started: `{drill['started_at']}`")
        lines.append(f"- Ended: `{drill['ended_at']}`")
        lines.append("```json")
        lines.append(json.dumps(drill["details"], indent=2, sort_keys=True))
        lines.append("```")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_operations_drills(run_id: str) -> dict[str, Any]:
    started = _utc_now_iso()
    drills = [
        _run_auth_abuse_drill(),
        _run_budget_contraction_drill(),
        _run_rollback_drill(),
    ]
    ended = _utc_now_iso()
    verdict = "pass" if all(d.passed for d in drills) else "fail"
    return {
        "run_id": run_id,
        "started_at": started,
        "ended_at": ended,
        "verdict": verdict,
        "drills": [asdict(item) for item in drills],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run v1 operations drills and write reports.")
    parser.add_argument(
        "--run-id",
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="Stable run id suffix for generated artifacts.",
    )
    parser.add_argument(
        "--out-dir",
        default="reports/release_gate",
        help="Directory for generated drill artifacts.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = run_operations_drills(run_id=args.run_id)
    json_path = out_dir / f"operations_drill_{args.run_id}.json"
    md_path = out_dir / f"operations_drill_{args.run_id}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown_report(md_path, report)

    print(json.dumps({"ok": True, "json_report": str(json_path), "md_report": str(md_path), "verdict": report["verdict"]}))
    return 0 if report["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
