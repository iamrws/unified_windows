"""Run a real CUDA transfer proof using shared AstraWeave runtime primitives."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from astrawave.cuda_runtime import run_cuda_transfer


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CUDA driver transfer PoC")
    parser.add_argument("--bytes", type=int, default=1_048_576, help="Transfer size in bytes")
    parser.add_argument("--device-index", type=int, default=0, help="CUDA device ordinal")
    parser.add_argument("--hold-ms", type=int, default=75, help="Delay after allocation before sampling NVML")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_cuda_transfer(
        size_bytes=args.bytes,
        device_index=args.device_index,
        hold_ms=max(args.hold_ms, 0),
    )
    if args.pretty:
        print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    else:
        print(json.dumps(result, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
