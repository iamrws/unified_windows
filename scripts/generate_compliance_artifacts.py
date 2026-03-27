"""Generate release governance artifacts for AstraWeave RC builds.

Artifacts:
- CycloneDX JSON SBOM
- third-party attribution manifest
- signing verification report
- checksums for generated artifacts
- consolidated compliance manifest
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _discover_imports(paths: list[Path]) -> set[str]:
    imports: set[str] = set()
    for py_path in paths:
        try:
            tree = ast.parse(py_path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if getattr(node, "level", 0):
                    continue
                if node.module:
                    imports.add(node.module.split(".")[0])
    return imports


def _runtime_python_files() -> list[Path]:
    candidates = sorted((REPO_ROOT / "astrawave").rglob("*.py"))
    return [p for p in candidates if "__pycache__" not in p.parts]


def _resolve_third_party_modules(imports: set[str]) -> list[dict[str, Any]]:
    stdlib = getattr(sys, "stdlib_module_names", set())
    package_map = importlib.metadata.packages_distributions()

    third_party = sorted(
        name
        for name in imports
        if name
        and name not in stdlib
        and name not in {"astrawave", "__future__"}
    )

    rows: list[dict[str, Any]] = []
    for module in third_party:
        distributions = package_map.get(module, [])
        if distributions:
            for dist_name in distributions:
                try:
                    md = importlib.metadata.metadata(dist_name)
                    version = importlib.metadata.version(dist_name)
                except importlib.metadata.PackageNotFoundError:
                    md = {}
                    version = "unknown"
                rows.append(
                    {
                        "module": module,
                        "distribution": dist_name,
                        "version": version,
                        "license": md.get("License", "UNKNOWN") or "UNKNOWN",
                    }
                )
        else:
            rows.append(
                {
                    "module": module,
                    "distribution": None,
                    "version": "unknown",
                    "license": "UNKNOWN",
                }
            )
    return rows


def _cyclonedx_component_ref(name: str, version: str) -> str:
    return f"pkg:pypi/{name}@{version}"


def _build_sbom(third_party: list[dict[str, Any]]) -> dict[str, Any]:
    root_ref = _cyclonedx_component_ref("astrawave", "0.1.0-rc")
    components = []
    dependencies = []

    for dep in third_party:
        dist = dep["distribution"] or dep["module"]
        version = dep["version"] or "unknown"
        bom_ref = _cyclonedx_component_ref(dist, version)
        components.append(
            {
                "type": "library",
                "name": dist,
                "version": version,
                "bom-ref": bom_ref,
                "licenses": [
                    {
                        "license": {
                            "name": dep["license"] or "UNKNOWN",
                        }
                    }
                ],
                "properties": [
                    {"name": "astrawave.module", "value": dep["module"]},
                ],
            }
        )
        dependencies.append(bom_ref)

    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{uuid.uuid4()}",
        "version": 1,
        "metadata": {
            "timestamp": _utc_now_iso(),
            "tools": [
                {
                    "vendor": "AstraWeave",
                    "name": "generate_compliance_artifacts.py",
                    "version": "1",
                }
            ],
            "component": {
                "type": "application",
                "name": "astrawave",
                "version": "0.1.0-rc",
                "bom-ref": root_ref,
                "licenses": [
                    {
                        "license": {
                            "id": "Apache-2.0",
                        }
                    }
                ],
            },
        },
        "components": components,
        "dependencies": [
            {
                "ref": root_ref,
                "dependsOn": dependencies,
            }
        ],
    }
    return sbom


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def generate_compliance_bundle(run_id: str, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    imports = _discover_imports(_runtime_python_files())
    third_party = _resolve_third_party_modules(imports)
    sbom_payload = _build_sbom(third_party=third_party)

    sbom_path = out_dir / f"sbom_{run_id}.cdx.json"
    _write_json(sbom_path, sbom_payload)

    attribution_payload = {
        "run_id": run_id,
        "generated_at": _utc_now_iso(),
        "project_license_target": "Apache-2.0",
        "runtime_scope": "astrawave package imports",
        "third_party_dependencies": third_party,
        "notes": (
            [
                "No non-stdlib runtime dependencies detected in astrawave package imports."
            ]
            if not third_party
            else []
        ),
    }
    attribution_path = out_dir / f"attribution_{run_id}.json"
    _write_json(attribution_path, attribution_payload)

    signing_payload = {
        "run_id": run_id,
        "generated_at": _utc_now_iso(),
        "channel": "rc",
        "artifact_type": "source-only",
        "binary_artifacts_discovered": [],
        "signing_policy_for_binaries": "required",
        "verification_status": "not_applicable_for_source_only_rc",
        "verification_notes": [
            "No compiled binaries were generated in this RC evidence bundle.",
            "If binary artifacts are later produced, code-sign verification becomes mandatory.",
        ],
    }
    signing_path = out_dir / f"signing_verification_{run_id}.json"
    _write_json(signing_path, signing_payload)

    checksums_payload = {
        "run_id": run_id,
        "generated_at": _utc_now_iso(),
        "algorithm": "sha256",
        "files": [
            {"path": str(sbom_path), "sha256": _sha256(sbom_path)},
            {"path": str(attribution_path), "sha256": _sha256(attribution_path)},
            {"path": str(signing_path), "sha256": _sha256(signing_path)},
        ],
    }
    checksums_path = out_dir / f"checksums_{run_id}.json"
    _write_json(checksums_path, checksums_payload)

    manifest_payload = {
        "run_id": run_id,
        "generated_at": _utc_now_iso(),
        "artifacts": {
            "sbom": str(sbom_path),
            "attribution": str(attribution_path),
            "signing_verification": str(signing_path),
            "checksums": str(checksums_path),
        },
        "summary": {
            "third_party_dependency_count": len(third_party),
            "source_only_rc": True,
        },
    }
    manifest_path = out_dir / f"compliance_manifest_{run_id}.json"
    _write_json(manifest_path, manifest_payload)
    return manifest_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate RC compliance artifacts.")
    parser.add_argument(
        "--run-id",
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="Stable run id suffix for generated artifacts.",
    )
    parser.add_argument(
        "--out-dir",
        default="reports/release_artifacts",
        help="Directory for generated compliance artifacts.",
    )
    args = parser.parse_args()

    out_dir = (REPO_ROOT / args.out_dir).resolve()
    manifest = generate_compliance_bundle(run_id=args.run_id, out_dir=out_dir)
    print(json.dumps({"ok": True, "manifest": manifest}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
