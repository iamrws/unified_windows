"""IPC client for AstraWeave service endpoints.

The client speaks the same request/response envelope shapes used by
`astrawave.ipc_protocol` and the local IPC server adapter. It uses
`multiprocessing.connection` for transport compatibility with both Windows
named pipes and local TCP sockets.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from ipaddress import ip_address
from multiprocessing.connection import Client, Connection
import logging
import os
from threading import Event, RLock, Thread
from typing import Any, Mapping, Protocol, cast
from urllib.parse import urlparse

from .errors import ApiError, ApiErrorCode
from .ipc_protocol import (
    CallerEnvelope,
    ErrorPayload,
    ErrorResponse,
    RequestEnvelope,
    SuccessResponse,
    api_error_from_payload,
    validate_response_payload,
)
from .security import CallerIdentity
from .types import MemoryTier, PolicyProfile, PressureSnapshot, ResidencySnapshot, ResidencyState, SessionState


_logger = logging.getLogger(__name__)

DEFAULT_TCP_ENDPOINT = "tcp://127.0.0.1:8765"
DEFAULT_PIPE_NAME = r"\\.\pipe\astrawave"
_POSITIVE_INT_OPTION_KEYS = {"num_ctx", "num_batch", "num_predict", "num_keep", "top_k"}
_NONNEGATIVE_INT_OPTION_KEYS = {"gpu_layers", "num_gpu"}
_NONNEGATIVE_FLOAT_OPTION_KEYS = {"temperature"}
_OPEN_INTERVAL_FLOAT_OPTION_KEYS = {"top_p", "repeat_penalty"}


def _authkey_from_env() -> bytes | None:
    value = os.environ.get("ASTRAWEAVE_IPC_AUTHKEY", "").strip()
    if not value:
        return None
    return value.encode("utf-8")


def _is_loopback_host(host: str) -> bool:
    normalized = host.strip().lower()
    if normalized == "localhost":
        return True
    try:
        return ip_address(normalized).is_loopback
    except ValueError:
        return False


def _require_loopback_host(host: str) -> str:
    normalized = host.strip() or "127.0.0.1"
    if not _is_loopback_host(normalized):
        raise ApiError(
            ApiErrorCode.INVALID_ARGUMENT,
            "IPC endpoint host must be localhost or loopback in v1",
        )
    return normalized


def _parse_port(value: int | str, *, allow_zero: bool = False) -> int:
    try:
        port = int(value)
    except (TypeError, ValueError) as exc:
        raise ApiError(ApiErrorCode.INVALID_ARGUMENT, f"invalid TCP port: {value}") from exc
    minimum = 0 if allow_zero else 1
    if port < minimum or port > 65535:
        bound = "0-65535" if allow_zero else "1-65535"
        raise ApiError(ApiErrorCode.INVALID_ARGUMENT, f"TCP port must be in range {bound}")
    return port


class _Transport(Protocol):
    def request(self, payload: dict[str, Any], timeout: float | None) -> dict[str, Any]:
        """Send one request and return a decoded response payload."""

    def close(self) -> None:
        """Close the underlying connection."""


def _normalize_caller(caller: CallerIdentity | None) -> CallerEnvelope | None:
    if caller is None:
        return None
    if not isinstance(caller, CallerIdentity):
        raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "caller must be a CallerIdentity or None")
    return CallerEnvelope(user_sid=caller.user_sid, pid=caller.pid)


def _jsonify(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, CallerIdentity):
        return {"user_sid": value.user_sid, "pid": value.pid}
    if is_dataclass(value):
        return {key: _jsonify(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


def _validate_tuning_profile(field_name: str, value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ApiError(ApiErrorCode.INVALID_ARGUMENT, f"{field_name} must be a non-empty string")
    return value.strip()


def _validate_tuning_options(field_name: str, options: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if options is None:
        return None
    if not isinstance(options, Mapping):
        raise ApiError(ApiErrorCode.INVALID_ARGUMENT, f"{field_name} must be a JSON object")

    normalized: dict[str, Any] = {}
    for raw_key, raw_value in options.items():
        if not isinstance(raw_key, str) or not raw_key.strip():
            raise ApiError(ApiErrorCode.INVALID_ARGUMENT, f"{field_name} keys must be non-empty strings")
        key = raw_key.strip()
        normalized[key] = _validate_tuning_option_value(field_name, key, raw_value)
    return normalized


def _validate_tuning_option_value(field_name: str, key: str, value: Any) -> Any:
    if key in _POSITIVE_INT_OPTION_KEYS:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ApiError(ApiErrorCode.INVALID_ARGUMENT, f"{field_name}.{key} must be a positive integer")
        return value
    if key in _NONNEGATIVE_INT_OPTION_KEYS:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ApiError(ApiErrorCode.INVALID_ARGUMENT, f"{field_name}.{key} must be a non-negative integer")
        return value
    if key in _NONNEGATIVE_FLOAT_OPTION_KEYS:
        if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) < 0.0:
            raise ApiError(ApiErrorCode.INVALID_ARGUMENT, f"{field_name}.{key} must be a non-negative number")
        return float(value)
    if key == "top_p":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ApiError(ApiErrorCode.INVALID_ARGUMENT, f"{field_name}.{key} must be a number")
        numeric_value = float(value)
        if numeric_value <= 0.0 or numeric_value > 1.0:
            raise ApiError(ApiErrorCode.INVALID_ARGUMENT, f"{field_name}.{key} must be between 0 and 1")
        return numeric_value
    if key in _OPEN_INTERVAL_FLOAT_OPTION_KEYS:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ApiError(ApiErrorCode.INVALID_ARGUMENT, f"{field_name}.{key} must be a number")
        numeric_value = float(value)
        if numeric_value <= 0.0:
            raise ApiError(ApiErrorCode.INVALID_ARGUMENT, f"{field_name}.{key} must be greater than 0")
        return numeric_value
    return _jsonify(value)


def _coerce_pressure_snapshot(value: Any) -> PressureSnapshot:
    if isinstance(value, PressureSnapshot):
        return value
    if not isinstance(value, Mapping):
        raise ApiError(ApiErrorCode.INTERNAL, "pressure snapshot response must be an object")
    return PressureSnapshot(
        session_id=str(value["session_id"]),
        vram_budget_bytes=int(value["vram_budget_bytes"]),
        vram_used_bytes=int(value["vram_used_bytes"]),
        pinned_ram_used_bytes=int(value["pinned_ram_used_bytes"]),
        pressure_level=float(value["pressure_level"]),
        policy_profile=PolicyProfile(str(value["policy_profile"])),
        timestamp_ms=int(value["timestamp_ms"]),
    )


def _coerce_residency_snapshot(value: Any) -> ResidencySnapshot:
    if isinstance(value, ResidencySnapshot):
        return value
    if not isinstance(value, Mapping):
        raise ApiError(ApiErrorCode.INTERNAL, "residency snapshot response must be an object")
    tensor_residency = {
        str(key): ResidencyState(str(item))
        for key, item in dict(value.get("tensor_residency", {})).items()
    }
    return ResidencySnapshot(
        session_id=str(value["session_id"]),
        session_state=SessionState(str(value["session_state"])),
        primary_tier=MemoryTier(str(value["primary_tier"])),
        tensor_residency=tensor_residency,
        vram_bytes=int(value["vram_bytes"]),
        pinned_ram_bytes=int(value["pinned_ram_bytes"]),
        pageable_ram_bytes=int(value["pageable_ram_bytes"]),
        cpu_only_bytes=int(value["cpu_only_bytes"]),
        active_run_in_progress=bool(value["active_run_in_progress"]),
    )


def _parse_endpoint(endpoint: str | None, prefer_named_pipe: bool) -> tuple[str, Any]:
    hint = (endpoint or "").strip()
    if not hint:
        if os.name == "nt" and prefer_named_pipe:
            return "AF_PIPE", DEFAULT_PIPE_NAME
        return "AF_INET", ("127.0.0.1", 8765)

    if hint == "auto":
        return _parse_endpoint(None, prefer_named_pipe)

    if hint.startswith("pipe://"):
        return "AF_PIPE", hint.removeprefix("pipe://")
    if hint.startswith("\\\\.\\pipe\\"):
        return "AF_PIPE", hint

    parsed = urlparse(hint)
    if parsed.scheme == "tcp":
        host = _require_loopback_host(parsed.hostname or "127.0.0.1")
        try:
            parsed_port = parsed.port
        except ValueError as exc:
            raise ApiError(ApiErrorCode.INVALID_ARGUMENT, f"invalid TCP endpoint: {hint}") from exc
        port = parsed_port if parsed_port is not None else 8765
        port = _parse_port(port)
        return "AF_INET", (host, port)
    if parsed.scheme in {"socket", "http", "https"}:
        host = _require_loopback_host(parsed.hostname or "127.0.0.1")
        try:
            parsed_port = parsed.port
        except ValueError as exc:
            raise ApiError(ApiErrorCode.INVALID_ARGUMENT, f"invalid TCP endpoint: {hint}") from exc
        port = parsed_port if parsed_port is not None else 8765
        port = _parse_port(port)
        return "AF_INET", (host, port)

    if ":" in hint and not hint.startswith("["):
        host, _, port_text = hint.rpartition(":")
        if not host:
            host = "127.0.0.1"
        host = _require_loopback_host(host)
        return "AF_INET", (host, _parse_port(port_text))

    if os.name == "nt" and prefer_named_pipe:
        return "AF_PIPE", hint

    return "AF_INET", (_require_loopback_host(hint), 8765)


class _ConnectionTransport:
    """Transport adapter over multiprocessing connection objects."""

    def __init__(
        self,
        family: str,
        address: Any,
        *,
        timeout: float | None,
        authkey: bytes | None = None,
    ) -> None:
        self._family = family
        self._address = address
        self._timeout = timeout
        self._authkey = authkey
        self._connection: Connection | None = None
        self._lock = RLock()

    def connect(self) -> None:
        with self._lock:
            if self._connection is not None:
                return
            self._connection = Client(self._address, family=self._family, authkey=self._authkey)

    def request(self, payload: dict[str, Any], timeout: float | None) -> dict[str, Any]:
        connection = self._require_connection()
        with self._lock:
            if timeout is None:
                return self._roundtrip(connection, payload)

            result_box: dict[str, Any] = {}
            error_box: list[BaseException] = []
            finished = Event()

            def worker() -> None:
                try:
                    result_box["value"] = self._roundtrip(connection, payload)
                except BaseException as exc:  # pragma: no cover - worker failure path
                    error_box.append(exc)
                finally:
                    finished.set()

            thread = Thread(target=worker, name="AstraWeaveIpcClientRequest", daemon=True)
            thread.start()
            if not finished.wait(timeout):
                self.close()
                # Give the worker thread a short grace period to finish
                # after the connection is closed, to avoid leaking it.
                thread.join(timeout=1.0)
                if thread.is_alive():
                    logging.getLogger(__name__).warning(
                        "IPC worker thread still alive after timeout; "
                        "thread %s may be leaked",
                        thread.name,
                    )
                raise ApiError(ApiErrorCode.TIMEOUT, "IPC request timed out")
            thread.join()
            if error_box:
                exc = error_box[0]
                if isinstance(exc, ApiError):
                    raise exc
                raise ApiError(ApiErrorCode.INTERNAL, "IPC request failed") from exc
            return cast(dict[str, Any], result_box["value"])

    def close(self) -> None:
        with self._lock:
            if self._connection is None:
                return
            try:
                self._connection.close()
            finally:
                self._connection = None

    def _require_connection(self) -> Connection:
        # H12 fix: check connection state inside lock to prevent TOCTOU race
        with self._lock:
            if self._connection is None:
                raise ApiError(ApiErrorCode.INVALID_STATE, "client is not connected")
            return self._connection

    def _roundtrip(self, connection: Connection, payload: dict[str, Any]) -> dict[str, Any]:
        connection.send(payload)
        response = connection.recv()
        if isinstance(response, Mapping):
            return cast(dict[str, Any], dict(response))
        if hasattr(response, "to_dict"):
            return cast(dict[str, Any], response.to_dict())
        raise ApiError(ApiErrorCode.INTERNAL, "response envelope must be an object")


class AstraWeaveIpcClient:
    """High-level IPC client for AstraWeave service APIs."""

    def __init__(
        self,
        endpoint: str | None = None,
        *,
        timeout: float | None = 30.0,
        default_caller: CallerIdentity | None = None,
        prefer_named_pipe: bool = True,
        authkey: bytes | None = None,
        transport_policy: str = "prefer_pipe",
    ) -> None:
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive or None")
        if transport_policy not in {"prefer_pipe", "pipe_only", "tcp_only"}:
            raise ValueError("transport_policy must be 'prefer_pipe', 'pipe_only', or 'tcp_only'")
        self.endpoint = endpoint or os.environ.get("ASTRAWEAVE_IPC_ENDPOINT") or "auto"
        self.timeout = timeout
        self.default_caller = default_caller
        self.prefer_named_pipe = prefer_named_pipe
        self.authkey = authkey if authkey is not None else _authkey_from_env()
        self.transport_policy = transport_policy
        self._transport: _Transport | None = None
        self._request_counter = 0
        self._counter_lock = RLock()  # M16 fix: thread-safe request counter

    @property
    def is_connected(self) -> bool:
        """Return whether a transport connection is active."""

        return self._transport is not None

    def connect(self) -> "AstraWeaveIpcClient":
        """Connect the client to the configured transport."""

        if self._transport is not None:
            return self

        family, address = _parse_endpoint(self.endpoint, self.prefer_named_pipe)
        try:
            transport = _ConnectionTransport(
                family,
                address,
                timeout=self.timeout,
                authkey=self.authkey,
            )
            transport.connect()
        except OSError as exc:
            if family == "AF_PIPE":
                # AUD-003 fix: respect transport_policy instead of silent fallback.
                if self.transport_policy == "pipe_only":
                    raise ApiError(
                        ApiErrorCode.INTERNAL,
                        "named pipe transport required by policy but connection failed",
                    ) from exc
                _logger.warning(
                    "Named pipe connection failed (%s); falling back to TCP 127.0.0.1:8765",
                    exc,
                )
                fallback = _ConnectionTransport(
                    "AF_INET",
                    ("127.0.0.1", 8765),
                    timeout=self.timeout,
                    authkey=self.authkey,
                )
                try:
                    fallback.connect()
                except OSError as fallback_exc:
                    raise ApiError(ApiErrorCode.INTERNAL, "failed to connect to AstraWeave IPC endpoint") from fallback_exc
                transport = fallback
            else:
                raise ApiError(ApiErrorCode.INTERNAL, "failed to connect to AstraWeave IPC endpoint") from exc

        self._transport = transport
        return self

    def close(self) -> None:
        """Close the active transport if one exists."""

        if self._transport is None:
            return
        try:
            self._transport.close()
        finally:
            self._transport = None

    def __enter__(self) -> "AstraWeaveIpcClient":
        return self.connect()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def call(
        self,
        method: str,
        params: Any | None = None,
        caller: CallerIdentity | None = None,
    ) -> Any:
        """Send one method call and return the decoded result."""

        if not method or not method.strip():
            raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "method must be a non-empty string")

        transport = self._ensure_transport()
        request = RequestEnvelope(
            id=self._next_request_id(),
            method=method,
            params=_jsonify(params if params is not None else {}),
            caller=_normalize_caller(caller or self.default_caller),
        )
        try:
            response_payload = transport.request(request.to_dict(), self.timeout)
        except ApiError:
            raise
        except Exception as exc:
            raise ApiError(ApiErrorCode.INTERNAL, f"IPC transport failure while calling {method}") from exc

        response = validate_response_payload(response_payload)
        if isinstance(response, SuccessResponse):
            return response.result
        if isinstance(response, ErrorResponse):
            raise response.error.to_api_error()
        raise ApiError(ApiErrorCode.INTERNAL, "unknown IPC response type")

    def CreateSession(self, caller: CallerIdentity | None = None) -> str:
        """Create a new session and return its opaque identifier."""

        return cast(str, self.call("CreateSession", {}, caller))

    def LoadModel(
        self,
        session_id: str,
        model_name: str,
        caller: CallerIdentity | None = None,
        *,
        runtime_backend: str | None = None,
        runtime_profile: str | None = None,
        runtime_backend_options: Mapping[str, Any] | None = None,
    ) -> None:
        params: dict[str, Any] = {"session_id": session_id, "model_name": model_name}
        if runtime_backend is not None:
            params["runtime_backend"] = runtime_backend
        normalized_profile = _validate_tuning_profile("runtime_profile", runtime_profile)
        if normalized_profile is not None:
            params["runtime_profile"] = normalized_profile
        normalized_options = _validate_tuning_options("runtime_backend_options", runtime_backend_options)
        if normalized_options is not None:
            params["runtime_backend_options"] = normalized_options
        self.call("LoadModel", params, caller)

    def RegisterTensor(
        self,
        session_id: str,
        tensor_name: str,
        size_bytes: int,
        caller: CallerIdentity | None = None,
    ) -> None:
        self.call(
            "RegisterTensor",
            {"session_id": session_id, "tensor_name": tensor_name, "size_bytes": size_bytes},
            caller,
        )

    def SetTierHint(
        self,
        session_id: str,
        tensor_name: str,
        tier: MemoryTier,
        caller: CallerIdentity | None = None,
    ) -> None:
        self.call(
            "SetTierHint",
            {"session_id": session_id, "tensor_name": tensor_name, "tier": tier.value},
            caller,
        )

    def PrefetchPlan(self, session_id: str, caller: CallerIdentity | None = None) -> list[dict[str, Any]]:
        result = self.call("PrefetchPlan", {"session_id": session_id}, caller)
        return cast(list[dict[str, Any]], result)

    def RunStep(
        self,
        session_id: str,
        step_name: str = "run",
        caller: CallerIdentity | None = None,
        *,
        prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        runtime_profile_override: str | None = None,
        runtime_backend_options_override: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"session_id": session_id, "step_name": step_name}
        if prompt is not None:
            params["prompt"] = prompt
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if temperature is not None:
            params["temperature"] = temperature
        normalized_profile = _validate_tuning_profile("runtime_profile_override", runtime_profile_override)
        if normalized_profile is not None:
            params["runtime_profile_override"] = normalized_profile
        normalized_options = _validate_tuning_options(
            "runtime_backend_options_override",
            runtime_backend_options_override,
        )
        if normalized_options is not None:
            params["runtime_backend_options_override"] = normalized_options
        return cast(dict[str, Any], self.call("RunStep", params, caller))

    def GetResidency(self, session_id: str, caller: CallerIdentity | None = None) -> ResidencySnapshot:
        return _coerce_residency_snapshot(self.call("GetResidency", {"session_id": session_id}, caller))

    def GetPressure(self, session_id: str, caller: CallerIdentity | None = None) -> PressureSnapshot:
        return _coerce_pressure_snapshot(self.call("GetPressure", {"session_id": session_id}, caller))

    def SetPolicy(
        self,
        session_id: str,
        policy: PolicyProfile,
        caller: CallerIdentity | None = None,
    ) -> None:
        self.call("SetPolicy", {"session_id": session_id, "policy": policy.value}, caller)

    def CloseSession(self, session_id: str, caller: CallerIdentity | None = None) -> None:
        self.call("CloseSession", {"session_id": session_id}, caller)

    def _ensure_transport(self) -> _Transport:
        if self._transport is None:
            self.connect()
        if self._transport is None:
            raise ApiError(ApiErrorCode.INTERNAL, "client failed to establish transport")
        return self._transport

    def _next_request_id(self) -> str:
        # M16 fix: use lock for thread-safe counter increment
        with self._counter_lock:
            self._request_counter += 1
            return f"req-{self._request_counter:08d}"


__all__ = [
    "AstraWeaveIpcClient",
    "DEFAULT_PIPE_NAME",
    "DEFAULT_TCP_ENDPOINT",
]
