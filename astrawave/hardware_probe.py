"""Real hardware probe helpers for Windows/NVIDIA environments.

The probe is intentionally fail-soft: missing tools, missing NVML, or runtime
errors are reflected in structured result blocks instead of bubbling up.
"""

from __future__ import annotations

import csv
import ctypes
import os
import subprocess
from dataclasses import dataclass
from ctypes import byref, c_uint, c_void_p, create_string_buffer
from ctypes.util import find_library
from time import time_ns
from typing import Any, Iterable

NVML_SUCCESS = 0
DEFAULT_COMMAND_TIMEOUT_SECONDS = 10.0
STRING_BUFFER_SIZE = 96


@dataclass(frozen=True, slots=True)
class _ProbeError:
    code: str
    message: str
    detail: str | None = None

    def as_dict(self) -> dict[str, str]:
        data: dict[str, str] = {"code": self.code, "message": self.message}
        if self.detail:
            data["detail"] = self.detail
        return data


def collect_hardware_probe() -> dict[str, Any]:
    """Collect the current NVIDIA hardware view in a JSON-serializable dict."""

    warnings: list[str] = []
    timestamp_ms = time_ns() // 1_000_000

    nvidia_smi_block = _probe_nvidia_smi(warnings)
    nvml_block = _probe_nvml(warnings)
    effective = _merge_effective_summary(nvidia_smi_block, nvml_block, warnings)

    return {
        "timestamp_ms": timestamp_ms,
        "nvidia_smi": nvidia_smi_block,
        "nvml": nvml_block,
        "effective": effective,
        "warnings": warnings,
    }


def _probe_nvidia_smi(warnings: list[str]) -> dict[str, Any]:
    nvidia_smi_path = os.environ.get("ASTRAWEAVE_NVIDIA_SMI_PATH", "nvidia-smi")
    command = [
        nvidia_smi_path,
        "--query-gpu=index,name,memory.total,memory.used,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ]
    block: dict[str, Any] = {
        "available": False,
        "status": "unavailable",
        "command": command,
        "driver_version": None,
        "devices": [],
        "error": None,
    }

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=DEFAULT_COMMAND_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as exc:
        block["error"] = _error_dict(
            "HARDWARE_PROBE_TOOL_MISSING",
            "nvidia-smi is not available on PATH",
            str(exc),
        )
        warnings.append("nvidia-smi unavailable")
        return block
    except subprocess.TimeoutExpired as exc:
        block["status"] = "timeout"
        block["error"] = _error_dict(
            "HARDWARE_PROBE_TIMEOUT",
            "nvidia-smi timed out while collecting hardware data",
            str(exc),
        )
        warnings.append("nvidia-smi timed out")
        return block
    except Exception as exc:  # pragma: no cover - defensive fallback
        block["status"] = "error"
        block["error"] = _error_dict(
            "HARDWARE_PROBE_INTERNAL",
            "unexpected failure while invoking nvidia-smi",
            repr(exc),
        )
        warnings.append("nvidia-smi probe failed")
        return block

    block["exit_code"] = completed.returncode
    block["stdout"] = completed.stdout.strip()
    block["stderr"] = completed.stderr.strip() or None

    if completed.returncode != 0:
        block["status"] = "error"
        block["error"] = _error_dict(
            "HARDWARE_PROBE_TOOL_ERROR",
            "nvidia-smi returned a non-zero exit code",
            completed.stderr.strip() or completed.stdout.strip() or None,
        )
        warnings.append("nvidia-smi returned an error")
        return block

    devices: list[dict[str, Any]] = []
    driver_version: str | None = None
    parse_errors: list[str] = []
    for row in _iter_csv_rows(completed.stdout):
        if len(row) < 6:
            parse_errors.append(f"unexpected nvidia-smi row shape: {row!r}")
            continue
        try:
            index = int(row[0].strip())
            name = row[1].strip()
            memory_total_mib = int(row[2].strip())
            memory_used_mib = int(row[3].strip())
            memory_free_mib = int(row[4].strip())
            row_driver_version = row[5].strip() or None
        except ValueError as exc:
            parse_errors.append(f"failed to parse nvidia-smi row {row!r}: {exc}")
            continue

        driver_version = driver_version or row_driver_version
        devices.append(
            {
                "index": index,
                "name": name,
                "memory_total_bytes": memory_total_mib * 1024 * 1024,
                "memory_used_bytes": memory_used_mib * 1024 * 1024,
                "memory_free_bytes": memory_free_mib * 1024 * 1024,
                "driver_version": row_driver_version,
            }
        )

    if parse_errors:
        warnings.extend(parse_errors)

    block["available"] = True
    if parse_errors and devices:
        block["status"] = "partial"
    elif parse_errors:
        block["status"] = "error"
    else:
        block["status"] = "ok" if devices else "empty"
    block["devices"] = devices
    block["driver_version"] = driver_version
    if parse_errors:
        block["error"] = _error_dict(
            "HARDWARE_PROBE_PARTIAL_PARSE",
            "nvidia-smi output contained rows that could not be parsed",
            "; ".join(parse_errors),
        )
    return block


def _probe_nvml(warnings: list[str]) -> dict[str, Any]:
    block: dict[str, Any] = {
        "available": False,
        "status": "unavailable",
        "library": None,
        "initialized": False,
        "driver_version": None,
        "devices": [],
        "error": None,
    }

    library = _load_nvml_library()
    if library is None:
        block["error"] = _error_dict(
            "HARDWARE_PROBE_LIBRARY_MISSING",
            "NVML library could not be loaded",
            "tried common NVML library names",
        )
        warnings.append("NVML library unavailable")
        return block

    block["library"] = library._name if getattr(library, "_name", None) else "nvml"
    api = _NvmlApi(library)
    try:
        api.init()
        block["initialized"] = True
        block["driver_version"] = api.system_driver_version()

        count = api.device_count()
        devices = []
        partial_read = False
        for index in range(count):
            device_entry: dict[str, Any] = {"index": index}
            try:
                handle = api.device_handle(index)
                if handle is None:
                    partial_read = True
                    warning = f"NVML skipped device index {index}"
                    warnings.append(warning)
                    device_entry["error"] = _error_dict(
                        "HARDWARE_PROBE_NVML_ERROR",
                        "NVML could not resolve a device handle",
                        f"index={index}",
                    )
                    devices.append(device_entry)
                    continue

                name = api.device_name(handle)
                memory = api.device_memory_info(handle)
                device_entry.update(
                    {
                        "name": name,
                        "memory_total_bytes": memory["total_bytes"],
                        "memory_used_bytes": memory["used_bytes"],
                        "memory_free_bytes": memory["free_bytes"],
                    }
                )
                devices.append(device_entry)
            except _NvmlError as exc:
                partial_read = True
                warnings.append(f"NVML device query failed for index {index}")
                device_entry["error"] = exc.as_dict()
                devices.append(device_entry)

        block["available"] = True
        if partial_read and devices:
            block["status"] = "partial"
        elif partial_read:
            block["status"] = "error"
        else:
            block["status"] = "ok" if devices else "empty"
        block["devices"] = devices
        if partial_read:
            block["error"] = _error_dict(
                "HARDWARE_PROBE_PARTIAL_READ",
                "NVML initialized but some device data could not be collected",
                f"device_count={count}",
            )
        return block
    except _NvmlError as exc:
        block["status"] = "error"
        block["error"] = exc.as_dict()
        warnings.append("NVML probe failed")
        return block
    except Exception as exc:  # pragma: no cover - defensive fallback
        block["status"] = "error"
        block["error"] = _error_dict(
            "HARDWARE_PROBE_INTERNAL",
            "unexpected failure while probing NVML",
            repr(exc),
        )
        warnings.append("NVML probe failed")
        return block
    finally:
        with _suppress_exception():
            api.shutdown()


def _merge_effective_summary(
    nvidia_smi_block: dict[str, Any],
    nvml_block: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    effective_devices = _merge_device_views(nvidia_smi_block.get("devices", []), nvml_block.get("devices", []))
    nvidia_valid_count = _count_valid_devices(nvidia_smi_block.get("devices", []))
    nvml_valid_count = _count_valid_devices(nvml_block.get("devices", []))
    if nvml_valid_count > 0 and nvml_valid_count >= nvidia_valid_count and nvml_block.get("available"):
        source = "nvml"
    elif nvidia_valid_count > 0 and nvidia_smi_block.get("available"):
        source = "nvidia-smi"
    elif nvml_block.get("available"):
        source = "nvml"
    elif nvidia_smi_block.get("available"):
        source = "nvidia-smi"
    else:
        source = "none"
    if source == "none":
        warnings.append("no NVIDIA hardware data could be collected")

    return {
        "source": source,
        "device_count": _count_valid_devices(effective_devices),
        "driver_version": nvml_block.get("driver_version") or nvidia_smi_block.get("driver_version"),
        "devices": effective_devices,
        "has_nvidia_gpu": _count_valid_devices(effective_devices) > 0,
    }


def _merge_device_views(
    nvidia_smi_devices: Iterable[dict[str, Any]],
    nvml_devices: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}

    for device in nvidia_smi_devices:
        index = device.get("index")
        if isinstance(index, int):
            merged[index] = {"index": index, **device, "source": "nvidia-smi"}

    for device in nvml_devices:
        index = device.get("index")
        if not isinstance(index, int):
            continue
        current = merged.get(index, {"index": index})
        current.update(device)
        current["source"] = "nvml" if current.get("source") != "nvidia-smi" else "nvml+nvidia-smi"
        merged[index] = current

    return [merged[index] for index in sorted(merged)]


def _count_valid_devices(devices: Iterable[dict[str, Any]]) -> int:
    count = 0
    for device in devices:
        if any(key in device for key in ("memory_total_bytes", "memory_used_bytes", "memory_free_bytes", "name")):
            count += 1
    return count


def _iter_csv_rows(stdout: str) -> list[list[str]]:
    rows: list[list[str]] = []
    reader = csv.reader(line for line in stdout.splitlines() if line.strip())
    for row in reader:
        rows.append([column.strip() for column in row])
    return rows


def _error_dict(code: str, message: str, detail: str | None = None) -> dict[str, str]:
    return _ProbeError(code=code, message=message, detail=detail).as_dict()


def _load_nvml_library() -> ctypes.CDLL | None:
    candidates = (
        "nvml.dll",
        "nvml64.dll",
        "libnvidia-ml.so.1",
        "libnvidia-ml.so",
    )
    loader = getattr(ctypes, "WinDLL", ctypes.CDLL)

    for candidate in candidates:
        try:
            return loader(candidate)
        except OSError:
            continue

    found = find_library("nvidia-ml")
    if found:
        try:
            return loader(found)
        except OSError:
            return None
    return None


class _NvmlError(Exception):
    def __init__(self, code: int, message: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail

    def as_dict(self) -> dict[str, str]:
        detail = self.detail if self.detail is not None else f"nvml_return_code={self.code}"
        return _error_dict("HARDWARE_PROBE_NVML_ERROR", self.message, detail)


class _NvmlMemoryInfo(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_uint64),
        ("free", ctypes.c_uint64),
        ("used", ctypes.c_uint64),
    ]


class _NvmlApi:
    def __init__(self, library: ctypes.CDLL) -> None:
        self._library = library

    def init(self) -> None:
        self._call_noargs("nvmlInit_v2")

    def shutdown(self) -> None:
        self._call_noargs("nvmlShutdown")

    def device_count(self) -> int:
        count = c_uint()
        self._call("nvmlDeviceGetCount_v2", byref(count))
        return int(count.value)

    def device_handle(self, index: int) -> c_void_p | None:
        handle = c_void_p()
        rc = self._call(
            "nvmlDeviceGetHandleByIndex_v2",
            c_uint(index),
            byref(handle),
            allow_missing=True,
        )
        if rc != NVML_SUCCESS:
            return None
        return handle

    def device_name(self, handle: c_void_p) -> str:
        buffer = create_string_buffer(STRING_BUFFER_SIZE)
        self._call("nvmlDeviceGetName", handle, ctypes.addressof(buffer), c_uint(len(buffer)))
        return _decode_c_string(buffer.value)

    def device_memory_info(self, handle: c_void_p) -> dict[str, int]:
        info = _NvmlMemoryInfo()
        self._call("nvmlDeviceGetMemoryInfo", handle, byref(info))
        return {
            "total_bytes": int(info.total),
            "free_bytes": int(info.free),
            "used_bytes": int(info.used),
        }

    def system_driver_version(self) -> str | None:
        if not hasattr(self._library, "nvmlSystemGetDriverVersion"):
            return None
        buffer = create_string_buffer(STRING_BUFFER_SIZE)
        rc = self._call(
            "nvmlSystemGetDriverVersion",
            ctypes.addressof(buffer),
            c_uint(len(buffer)),
            allow_missing=True,
        )
        if rc != NVML_SUCCESS:
            return None
        return _decode_c_string(buffer.value)

    def _call_noargs(self, name: str) -> None:
        self._call(name)

    def _call(self, name: str, *args: Any, allow_missing: bool = False) -> int:
        func = getattr(self._library, name, None)
        if func is None:
            if allow_missing:
                return -1
            raise _NvmlError(-1, f"NVML function missing: {name}")

        func.restype = c_uint
        func.argtypes = self._argtypes_for(name)
        rc = int(func(*args))
        if rc != NVML_SUCCESS and not allow_missing:
            raise _NvmlError(rc, f"NVML call failed: {name}", f"nvml_return_code={rc}")
        return rc

    @staticmethod
    def _argtypes_for(name: str) -> list[Any] | None:
        mapping: dict[str, list[Any] | None] = {
            "nvmlInit_v2": None,
            "nvmlShutdown": None,
            "nvmlDeviceGetCount_v2": [ctypes.POINTER(c_uint)],
            "nvmlDeviceGetHandleByIndex_v2": [c_uint, ctypes.POINTER(c_void_p)],
            "nvmlDeviceGetName": [c_void_p, ctypes.c_void_p, c_uint],
            "nvmlDeviceGetMemoryInfo": [c_void_p, ctypes.POINTER(_NvmlMemoryInfo)],
            "nvmlSystemGetDriverVersion": [ctypes.c_void_p, c_uint],
        }
        return mapping.get(name)


class _suppress_exception:
    """Suppress Exception only; let BaseException (SystemExit, KeyboardInterrupt) propagate."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:  # H14 fix
        if exc_type is not None and not issubclass(exc_type, Exception):
            return False
        return True


def _decode_c_string(value: bytes) -> str:
    return value.decode("utf-8", errors="replace").rstrip("\x00").strip()
