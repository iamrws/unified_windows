"""CUDA runtime helpers for real host<->device transfer verification.

The functions in this module are dependency-light and use `ctypes` against
`nvcuda.dll` so they can run on Windows machines without Python CUDA packages.
"""

from __future__ import annotations

import ctypes
import os
import zlib
from time import perf_counter_ns, sleep, time_ns
from typing import Any, Iterable

try:  # pragma: no cover - optional in stripped builds
    from .hardware_probe import collect_hardware_probe
except Exception:  # pragma: no cover
    collect_hardware_probe = None

CUDA_SUCCESS = 0
DEFAULT_TRANSFER_BYTES = 1_048_576
DEFAULT_DEVICE_INDEX = 0
MAX_TRANSFER_BYTES = 1 * 1024 * 1024 * 1024  # M9: 1 GiB upper bound on buffer size

CUresult = ctypes.c_int
CUdevice = ctypes.c_int
CUcontext = ctypes.c_void_p
CUdeviceptr = ctypes.c_uint64


class CudaDriverError(RuntimeError):
    """Typed CUDA driver call failure."""

    def __init__(self, code: int, operation: str, detail: str) -> None:
        super().__init__(f"{operation} failed with CUDA error {code}: {detail}")
        self.code = code
        self.operation = operation
        self.detail = detail


def run_cuda_transfer(
    *,
    size_bytes: int = DEFAULT_TRANSFER_BYTES,
    device_index: int = DEFAULT_DEVICE_INDEX,
    hold_ms: int = 0,
) -> dict[str, Any]:
    """Run a real CUDA host->device->host transfer and return structured results."""

    if os.name != "nt":
        return _error_result(
            code="CUDA_POC_UNSUPPORTED_PLATFORM",
            message="This runtime currently supports Windows only.",
            size_bytes=size_bytes,
            device_index=device_index,
        )
    if size_bytes <= 0:
        return _error_result(
            code="CUDA_POC_INVALID_ARGUMENT",
            message="size_bytes must be positive",
            size_bytes=size_bytes,
            device_index=device_index,
        )
    # M9: reject unreasonably large buffer requests
    if size_bytes > MAX_TRANSFER_BYTES:
        return _error_result(
            code="CUDA_POC_INVALID_ARGUMENT",
            message=f"size_bytes ({size_bytes}) exceeds MAX_TRANSFER_BYTES ({MAX_TRANSFER_BYTES})",
            size_bytes=size_bytes,
            device_index=device_index,
        )

    try:
        library = ctypes.WinDLL("nvcuda.dll")
    except OSError as exc:
        return _error_result(
            code="CUDA_POC_DRIVER_MISSING",
            message="nvcuda.dll was not found",
            detail=str(exc),
            size_bytes=size_bytes,
            device_index=device_index,
        )

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
    # H10 fix: generate test pattern in bulk instead of byte-by-byte O(n) Python loop
    pattern_bytes = bytes((i * 131 + 17) % 251 for i in range(min(size_bytes, 4096)))
    full_pattern = (pattern_bytes * ((size_bytes // len(pattern_bytes)) + 1))[:size_bytes]
    host_src = (ctypes.c_ubyte * size_bytes).from_buffer_copy(full_pattern)
    host_dst = (ctypes.c_ubyte * size_bytes)()

    src_crc32 = f"{zlib.crc32(full_pattern) & 0xFFFFFFFF:08x}"
    started_ns = perf_counter_ns()
    timings_ms: dict[str, float] = {}
    nvml_before = _capture_nvml_snapshot(device_index)
    nvml_after_alloc = _capture_nvml_snapshot(device_index)
    nvml_after_copy = _capture_nvml_snapshot(device_index)
    op_error: dict[str, Any] | None = None

    try:
        _call_cuda(library, init_name, cu_init, 0)
        _call_cuda(library, device_get_name, cu_device_get, ctypes.byref(device), device_index)
        _call_cuda(library, ctx_create_name, cu_ctx_create, ctypes.byref(context), 0, device)
        _call_cuda(library, mem_alloc_name, cu_mem_alloc, ctypes.byref(device_ptr), size_bytes)

        if hold_ms > 0:
            sleep(hold_ms / 1000.0)
        nvml_after_alloc = _capture_nvml_snapshot(device_index)

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
        nvml_after_copy = _capture_nvml_snapshot(device_index)
    except CudaDriverError as exc:
        op_error = {
            "code": "CUDA_POC_DRIVER_ERROR",
            "message": str(exc),
            "cuda_error_code": exc.code,
            "operation": exc.operation,
        }

    # M8: robust cleanup that handles edge cases where device_ptr or context
    # may be truthy but hold invalid values after partial failure.
    cleanup_errors: list[str] = []
    if device_ptr.value and device_ptr.value != ctypes.c_uint64(-1).value:
        try:
            rc = cu_mem_free(device_ptr)
            if rc != CUDA_SUCCESS:
                cleanup_errors.append(f"cuMemFree returned CUDA error {rc}")
        except Exception as exc:  # pragma: no cover - cleanup best-effort
            cleanup_errors.append(f"cuMemFree cleanup failed: {exc}")
    if context and context.value is not None:
        try:
            rc = cu_ctx_destroy(context)
            if rc != CUDA_SUCCESS:
                cleanup_errors.append(f"cuCtxDestroy returned CUDA error {rc}")
        except Exception as exc:  # pragma: no cover - cleanup best-effort
            cleanup_errors.append(f"cuCtxDestroy cleanup failed: {exc}")

    nvml_after_free = _capture_nvml_snapshot(device_index)
    nvml_observation = _summarize_nvml_observation(
        before=nvml_before,
        after_alloc=nvml_after_alloc,
        after_copy=nvml_after_copy,
        after_free=nvml_after_free,
    )
    if cleanup_errors:
        nvml_observation["warnings"] = cleanup_errors

    if op_error is not None:
        return {
            "ok": False,
            "timestamp_ms": time_ns() // 1_000_000,
            "size_bytes": size_bytes,
            "device_index": device_index,
            "nvml_observation": nvml_observation,
            "error": op_error,
        }

    dst_crc32 = f"{zlib.crc32(bytes(host_dst)) & 0xFFFFFFFF:08x}"
    verified = src_crc32 == dst_crc32
    total_ms = (perf_counter_ns() - started_ns) / 1_000_000.0
    copy_h2d_ms = timings_ms.get("copy_h2d_ms", 0.0)
    copy_d2h_ms = timings_ms.get("copy_d2h_ms", 0.0)
    round_trip_gib_per_sec = 0.0
    if copy_h2d_ms + copy_d2h_ms > 0:
        seconds = (copy_h2d_ms + copy_d2h_ms) / 1000.0
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
            "copy_h2d_ms": copy_h2d_ms,
            "copy_d2h_ms": copy_d2h_ms,
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
        "nvml_observation": nvml_observation,
        "error": None if verified else {
            "code": "CUDA_POC_VERIFY_MISMATCH",
            "message": "round-trip bytes did not match",
        },
    }


def _summarize_nvml_observation(
    *,
    before: dict[str, Any],
    after_alloc: dict[str, Any],
    after_copy: dict[str, Any],
    after_free: dict[str, Any],
) -> dict[str, Any]:
    before_used = before.get("used_bytes")
    peak_used = _max_int(
        before_used,
        after_alloc.get("used_bytes"),
        after_copy.get("used_bytes"),
        after_free.get("used_bytes"),
    )
    observed_delta_bytes = None
    if isinstance(before_used, int) and isinstance(peak_used, int):
        observed_delta_bytes = max(0, peak_used - before_used)
    return {
        "device_index": before.get("device_index"),
        "source": before.get("source", "none"),
        "before": before,
        "after_alloc": after_alloc,
        "after_copy": after_copy,
        "after_free": after_free,
        "peak_used_bytes": peak_used,
        "observed_delta_bytes": observed_delta_bytes,
        "observed": bool(observed_delta_bytes and observed_delta_bytes > 0),
    }


def _capture_nvml_snapshot(device_index: int) -> dict[str, Any]:
    snapshot = {
        "timestamp_ms": time_ns() // 1_000_000,
        "device_index": device_index,
        "used_bytes": None,
        "source": "none",
        "error": None,
    }
    if collect_hardware_probe is None:
        snapshot["error"] = "hardware_probe module unavailable"
        return snapshot
    try:
        probe = collect_hardware_probe()
    except Exception as exc:  # pragma: no cover - defensive
        snapshot["error"] = str(exc)
        return snapshot

    nvml_block = probe.get("nvml")
    if isinstance(nvml_block, dict) and nvml_block.get("available"):
        devices = nvml_block.get("devices")
        if isinstance(devices, list):
            for entry in devices:
                if isinstance(entry, dict) and int(entry.get("index", -1)) == device_index:
                    value = entry.get("memory_used_bytes")
                    if isinstance(value, int):
                        snapshot["used_bytes"] = value
                        snapshot["source"] = "nvml"
                        return snapshot
    nvidia_block = probe.get("nvidia_smi")
    if isinstance(nvidia_block, dict) and nvidia_block.get("available"):
        devices = nvidia_block.get("devices")
        if isinstance(devices, list):
            for entry in devices:
                if isinstance(entry, dict) and int(entry.get("index", -1)) == device_index:
                    value = entry.get("memory_used_bytes")
                    if isinstance(value, int):
                        snapshot["used_bytes"] = value
                        snapshot["source"] = "nvidia-smi"
                        return snapshot

    snapshot["error"] = "no NVML/nvidia-smi memory reading for device index"
    return snapshot


def _max_int(*values: Any) -> int | None:
    ints = [item for item in values if isinstance(item, int)]
    return max(ints) if ints else None


def _error_result(
    *,
    code: str,
    message: str,
    size_bytes: int,
    device_index: int,
    detail: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": False,
        "timestamp_ms": time_ns() // 1_000_000,
        "size_bytes": size_bytes,
        "device_index": device_index,
        "error": {
            "code": code,
            "message": message,
        },
        "nvml_observation": {
            "device_index": device_index,
            "source": "none",
            "observed": False,
            "observed_delta_bytes": None,
        },
    }
    if detail:
        payload["error"]["detail"] = detail
    return payload


def _resolve_symbol(library: ctypes.WinDLL, names: Iterable[str]) -> tuple[Any, str]:
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
