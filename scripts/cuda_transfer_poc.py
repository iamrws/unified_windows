"""Real CUDA host<->device transfer proof-of-concept.

This script uses the CUDA Driver API via `ctypes` against `nvcuda.dll` to:
1) initialize CUDA,
2) create a context,
3) allocate a device buffer,
4) copy bytes host->device->host,
5) verify round-trip integrity.

It emits JSON so results can be archived in runlogs.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import zlib
from time import perf_counter_ns, time_ns
from typing import Any, Iterable

CUDA_SUCCESS = 0

CUresult = ctypes.c_int
CUdevice = ctypes.c_int
CUcontext = ctypes.c_void_p
CUdeviceptr = ctypes.c_uint64


class CudaDriverError(RuntimeError):
    def __init__(self, code: int, operation: str, detail: str) -> None:
        super().__init__(f"{operation} failed with CUDA error {code}: {detail}")
        self.code = code
        self.operation = operation
        self.detail = detail


def _resolve_symbol(library: ctypes.WinDLL, names: Iterable[str]):
    for name in names:
        symbol = getattr(library, name, None)
        if symbol is not None:
            return symbol, name
    available = ", ".join(names)
    raise AttributeError(f"none of the symbols are available: {available}")


def _lookup_cuda_error(library: ctypes.WinDLL, code: int) -> str:
    get_error_name = getattr(library, "cuGetErrorName", None)
    get_error_string = getattr(library, "cuGetErrorString", None)
    text_parts: list[str] = []

    if get_error_name is not None:
        get_error_name.restype = CUresult
        get_error_name.argtypes = [CUresult, ctypes.POINTER(ctypes.c_char_p)]
        name_ptr = ctypes.c_char_p()
        if int(get_error_name(CUresult(code), ctypes.byref(name_ptr))) == CUDA_SUCCESS and name_ptr.value:
            text_parts.append(name_ptr.value.decode("utf-8", errors="replace"))

    if get_error_string is not None:
        get_error_string.restype = CUresult
        get_error_string.argtypes = [CUresult, ctypes.POINTER(ctypes.c_char_p)]
        message_ptr = ctypes.c_char_p()
        if int(get_error_string(CUresult(code), ctypes.byref(message_ptr))) == CUDA_SUCCESS and message_ptr.value:
            text_parts.append(message_ptr.value.decode("utf-8", errors="replace"))

    return " | ".join(text_parts) if text_parts else "unknown CUDA error"


def _call_cuda(
    library: ctypes.WinDLL,
    operation: str,
    function,
    *args: Any,
) -> int:
    rc = int(function(*args))
    if rc != CUDA_SUCCESS:
        raise CudaDriverError(rc, operation, _lookup_cuda_error(library, rc))
    return rc


def run_transfer_poc(*, size_bytes: int, device_index: int) -> dict[str, Any]:
    if os.name != "nt":
        return {
            "ok": False,
            "error": {
                "code": "CUDA_POC_UNSUPPORTED_PLATFORM",
                "message": "This PoC currently supports Windows only.",
            },
        }

    if size_bytes <= 0:
        return {
            "ok": False,
            "error": {
                "code": "CUDA_POC_INVALID_ARGUMENT",
                "message": "size_bytes must be positive",
            },
        }

    try:
        library = ctypes.WinDLL("nvcuda.dll")
    except OSError as exc:
        return {
            "ok": False,
            "error": {
                "code": "CUDA_POC_DRIVER_MISSING",
                "message": "nvcuda.dll was not found",
                "detail": str(exc),
            },
        }

    cu_init, init_name = _resolve_symbol(library, ("cuInit",))
    cu_device_get, device_get_name = _resolve_symbol(library, ("cuDeviceGet",))
    cu_ctx_create, ctx_create_name = _resolve_symbol(library, ("cuCtxCreate_v2", "cuCtxCreate"))
    cu_ctx_destroy, ctx_destroy_name = _resolve_symbol(library, ("cuCtxDestroy_v2", "cuCtxDestroy"))
    cu_mem_alloc, mem_alloc_name = _resolve_symbol(library, ("cuMemAlloc_v2", "cuMemAlloc"))
    cu_mem_free, mem_free_name = _resolve_symbol(library, ("cuMemFree_v2", "cuMemFree"))
    cu_memcpy_htod, memcpy_htod_name = _resolve_symbol(library, ("cuMemcpyHtoD_v2", "cuMemcpyHtoD"))
    cu_memcpy_dtoh, memcpy_dtoh_name = _resolve_symbol(library, ("cuMemcpyDtoH_v2", "cuMemcpyDtoH"))

    cu_init.restype = CUresult
    cu_init.argtypes = [ctypes.c_uint]

    cu_device_get.restype = CUresult
    cu_device_get.argtypes = [ctypes.POINTER(CUdevice), ctypes.c_int]

    cu_ctx_create.restype = CUresult
    cu_ctx_create.argtypes = [ctypes.POINTER(CUcontext), ctypes.c_uint, CUdevice]

    cu_ctx_destroy.restype = CUresult
    cu_ctx_destroy.argtypes = [CUcontext]

    cu_mem_alloc.restype = CUresult
    cu_mem_alloc.argtypes = [ctypes.POINTER(CUdeviceptr), ctypes.c_size_t]

    cu_mem_free.restype = CUresult
    cu_mem_free.argtypes = [CUdeviceptr]

    cu_memcpy_htod.restype = CUresult
    cu_memcpy_htod.argtypes = [CUdeviceptr, ctypes.c_void_p, ctypes.c_size_t]

    cu_memcpy_dtoh.restype = CUresult
    cu_memcpy_dtoh.argtypes = [ctypes.c_void_p, CUdeviceptr, ctypes.c_size_t]

    device = CUdevice()
    context = CUcontext()
    device_ptr = CUdeviceptr(0)

    host_src = (ctypes.c_ubyte * size_bytes)()
    host_dst = (ctypes.c_ubyte * size_bytes)()
    for index in range(size_bytes):
        host_src[index] = (index * 131 + 17) % 251

    src_crc32 = f"{zlib.crc32(bytes(host_src)) & 0xFFFFFFFF:08x}"
    timings_ms: dict[str, float] = {}
    started_ns = perf_counter_ns()

    try:
        _call_cuda(library, init_name, cu_init, 0)
        _call_cuda(library, device_get_name, cu_device_get, ctypes.byref(device), device_index)
        _call_cuda(library, ctx_create_name, cu_ctx_create, ctypes.byref(context), 0, device)
        _call_cuda(library, mem_alloc_name, cu_mem_alloc, ctypes.byref(device_ptr), size_bytes)

        t0 = perf_counter_ns()
        _call_cuda(
            library,
            memcpy_htod_name,
            cu_memcpy_htod,
            device_ptr,
            ctypes.cast(host_src, ctypes.c_void_p),
            size_bytes,
        )
        timings_ms["copy_h2d_ms"] = (perf_counter_ns() - t0) / 1_000_000.0

        t1 = perf_counter_ns()
        _call_cuda(
            library,
            memcpy_dtoh_name,
            cu_memcpy_dtoh,
            ctypes.cast(host_dst, ctypes.c_void_p),
            device_ptr,
            size_bytes,
        )
        timings_ms["copy_d2h_ms"] = (perf_counter_ns() - t1) / 1_000_000.0

        dst_crc32 = f"{zlib.crc32(bytes(host_dst)) & 0xFFFFFFFF:08x}"
        verified = src_crc32 == dst_crc32
        total_ms = (perf_counter_ns() - started_ns) / 1_000_000.0
        round_trip_gib_per_sec = 0.0
        if timings_ms["copy_h2d_ms"] + timings_ms["copy_d2h_ms"] > 0:
            seconds = (timings_ms["copy_h2d_ms"] + timings_ms["copy_d2h_ms"]) / 1000.0
            round_trip_gib_per_sec = (2 * size_bytes) / seconds / (1024**3)

        return {
            "ok": verified,
            "timestamp_ms": time_ns() // 1_000_000,
            "size_bytes": size_bytes,
            "device_index": int(device.value),
            "cuda_symbols": {
                "ctx_create": ctx_create_name,
                "ctx_destroy": ctx_destroy_name,
                "mem_alloc": mem_alloc_name,
                "mem_free": mem_free_name,
                "memcpy_h2d": memcpy_htod_name,
                "memcpy_d2h": memcpy_dtoh_name,
            },
            "timings_ms": {
                "copy_h2d_ms": timings_ms["copy_h2d_ms"],
                "copy_d2h_ms": timings_ms["copy_d2h_ms"],
                "total_ms": total_ms,
            },
            "throughput": {
                "round_trip_gib_per_sec": round_trip_gib_per_sec,
            },
            "verification": {
                "src_crc32": src_crc32,
                "dst_crc32": dst_crc32,
                "matched": verified,
            },
            "error": None if verified else {
                "code": "CUDA_POC_VERIFY_MISMATCH",
                "message": "round-trip bytes did not match",
            },
        }
    except CudaDriverError as exc:
        return {
            "ok": False,
            "timestamp_ms": time_ns() // 1_000_000,
            "size_bytes": size_bytes,
            "device_index": device_index,
            "error": {
                "code": "CUDA_POC_DRIVER_ERROR",
                "message": str(exc),
                "cuda_error_code": exc.code,
                "operation": exc.operation,
            },
        }
    finally:
        if device_ptr.value:
            try:
                cu_mem_free(device_ptr)
            except Exception:
                pass
        if context:
            try:
                cu_ctx_destroy(context)
            except Exception:
                pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CUDA driver transfer PoC")
    parser.add_argument("--bytes", type=int, default=1_048_576, help="Transfer size in bytes")
    parser.add_argument("--device-index", type=int, default=0, help="CUDA device ordinal")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_transfer_poc(size_bytes=args.bytes, device_index=args.device_index)
    if args.pretty:
        print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    else:
        print(json.dumps(result, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
