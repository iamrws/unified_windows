"""Run hardware execution gate checks for service-triggered CUDA RunStep mode."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from astrawave.hardware_probe import collect_hardware_probe
from astrawave.security import CallerIdentity
from astrawave.service import DEFAULT_SERVICE_OWNER_SID, AstraWeaveService
from astrawave.types import MemoryTier


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Hardware Gate Report",
        "",
        f"- Run id: `{payload['run_id']}`",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Verdict: `{payload['verdict']}`",
        "",
        "| Gate | Result | Note |",
        "| --- | --- | --- |",
    ]
    for gate, check in payload.get("gate_checks", {}).items():
        lines.append(f"| `{gate}` | `{'pass' if check.get('passed') else 'fail'}` | {check.get('note', 'n/a')} |")
    lines.extend(
        [
            "",
            "## Runtime Result",
            "",
            "```json",
            json.dumps(payload.get("runtime_result", {}), indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _probe_used_bytes(probe: dict[str, Any], *, device_index: int) -> int | None:
    nvml = probe.get("nvml")
    if isinstance(nvml, dict):
        devices = nvml.get("devices")
        if isinstance(devices, list):
            for device in devices:
                if isinstance(device, dict) and int(device.get("index", -1)) == device_index:
                    value = device.get("memory_used_bytes")
                    if isinstance(value, int):
                        return value
    nvidia_smi = probe.get("nvidia_smi")
    if isinstance(nvidia_smi, dict):
        devices = nvidia_smi.get("devices")
        if isinstance(devices, list):
            for device in devices:
                if isinstance(device, dict) and int(device.get("index", -1)) == device_index:
                    value = device.get("memory_used_bytes")
                    if isinstance(value, int):
                        return value
    return None


def run_hardware_gate(*, run_id: str, transfer_bytes: int, device_index: int, hold_ms: int) -> dict[str, Any]:
    owner = CallerIdentity(user_sid=DEFAULT_SERVICE_OWNER_SID, pid=os.getpid())
    service = AstraWeaveService(runstep_mode="hardware")

    before_probe = collect_hardware_probe()
    before_used = _probe_used_bytes(before_probe, device_index=device_index)

    session_id = service.CreateSession(caller_identity=owner)
    service.LoadModel(session_id, "hardware-gate-model", caller_identity=owner)
    service.RegisterTensor(session_id, "gate_tensor", max(transfer_bytes, 1), caller_identity=owner)
    service.SetTierHint(session_id, "gate_tensor", MemoryTier.HOT, caller_identity=owner)

    # Ensure service request context mirrors this gate run regardless of environment defaults.
    previous_bytes = None
    previous_device = None
    previous_hold = None
    try:
        import os

        previous_bytes = os.environ.get("ASTRAWEAVE_HARDWARE_TRANSFER_BYTES")
        previous_device = os.environ.get("ASTRAWEAVE_HARDWARE_DEVICE_INDEX")
        previous_hold = os.environ.get("ASTRAWEAVE_HARDWARE_HOLD_MS")
        os.environ["ASTRAWEAVE_HARDWARE_TRANSFER_BYTES"] = str(max(transfer_bytes, 1))
        os.environ["ASTRAWEAVE_HARDWARE_DEVICE_INDEX"] = str(max(device_index, 0))
        os.environ["ASTRAWEAVE_HARDWARE_HOLD_MS"] = str(max(hold_ms, 0))
        runtime_result = service.RunStep(session_id, step_name="hardware_gate", caller_identity=owner)
    finally:
        import os

        if previous_bytes is None:
            os.environ.pop("ASTRAWEAVE_HARDWARE_TRANSFER_BYTES", None)
        else:
            os.environ["ASTRAWEAVE_HARDWARE_TRANSFER_BYTES"] = previous_bytes
        if previous_device is None:
            os.environ.pop("ASTRAWEAVE_HARDWARE_DEVICE_INDEX", None)
        else:
            os.environ["ASTRAWEAVE_HARDWARE_DEVICE_INDEX"] = previous_device
        if previous_hold is None:
            os.environ.pop("ASTRAWEAVE_HARDWARE_HOLD_MS", None)
        else:
            os.environ["ASTRAWEAVE_HARDWARE_HOLD_MS"] = previous_hold

    after_probe = collect_hardware_probe()
    after_used = _probe_used_bytes(after_probe, device_index=device_index)

    hardware_result = runtime_result.get("hardware_result")
    if not isinstance(hardware_result, dict):
        hardware_result = {}
    nvml_observation = hardware_result.get("nvml_observation")
    if not isinstance(nvml_observation, dict):
        nvml_observation = {}

    run_mode_ok = runtime_result.get("run_mode") == "hardware"
    transfer_ok = bool(hardware_result.get("ok"))
    observed_delta = nvml_observation.get("observed_delta_bytes")
    delta_observed = bool(nvml_observation.get("observed"))
    probe_delta = None
    if isinstance(before_used, int) and isinstance(after_used, int):
        probe_delta = after_used - before_used

    gate_checks = {
        "service_hardware_mode_active": {
            "passed": run_mode_ok,
            "note": f"run_mode={runtime_result.get('run_mode')}",
        },
        "service_hardware_transfer_ok": {
            "passed": transfer_ok,
            "note": f"hardware_result.ok={hardware_result.get('ok')}",
        },
        "service_triggered_nvml_delta_observed": {
            "passed": delta_observed,
            "note": f"observed_delta_bytes={observed_delta}",
        },
    }
    verdict = "pass" if all(item["passed"] for item in gate_checks.values()) else "fail"

    return {
        "run_id": run_id,
        "generated_at": _utc_now_iso(),
        "verdict": verdict,
        "config": {
            "transfer_bytes": transfer_bytes,
            "device_index": device_index,
            "hold_ms": hold_ms,
        },
        "gate_checks": gate_checks,
        "runtime_result": runtime_result,
        "probe_snapshot": {
            "before_used_bytes": before_used,
            "after_used_bytes": after_used,
            "after_minus_before_bytes": probe_delta,
        },
        "artifacts": {
            "before_probe": before_probe,
            "after_probe": after_probe,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run hardware release-gate checks.")
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    parser.add_argument("--out-dir", default="reports/release_gate")
    parser.add_argument("--transfer-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--hold-ms", type=int, default=250)
    args = parser.parse_args()

    out_dir = (REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = run_hardware_gate(
        run_id=args.run_id,
        transfer_bytes=max(args.transfer_bytes, 1),
        device_index=max(args.device_index, 0),
        hold_ms=max(args.hold_ms, 0),
    )

    json_path = out_dir / f"hardware_gate_{args.run_id}.json"
    md_path = out_dir / f"hardware_gate_{args.run_id}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(md_path, report)

    print(
        json.dumps(
            {
                "ok": True,
                "json_report": str(json_path),
                "md_report": str(md_path),
                "verdict": report["verdict"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
