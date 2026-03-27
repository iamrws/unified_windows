"""Local IPC server adapter for AstraWeave.

This module intentionally uses `astrawave.ipc_protocol` as the canonical wire
contract. Compatibility wrappers remain for older imports, but request/response
validation and envelope semantics are centralized in the shared protocol layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from inspect import Parameter, signature
from ipaddress import ip_address
from multiprocessing.connection import Listener
import os
from threading import Event, RLock, Thread
from typing import Any, Callable, Mapping

from .errors import ApiError, ApiErrorCode
from .ipc_protocol import (
    CallerEnvelope,
    ErrorPayload,
    ErrorResponse,
    MAX_CONTROL_PLANE_PAYLOAD_BYTES,
    RequestEnvelope,
    SuccessResponse,
    estimate_json_size_bytes,
    error_response,
    success_response,
    to_json_value,
    validate_request_payload,
)
from .security import CallerIdentity, SecurityDecision, attest_caller_identity
from .service import AstraWeaveService
from .telemetry import SecurityEvent, TelemetryReasonCode
from .types import MemoryTier, PolicyProfile

__all__ = [
    "IpcRequestEnvelope",
    "IpcResponseEnvelope",
    "IpcSuccessEnvelope",
    "IpcErrorEnvelope",
    "IpcProtocolError",
    "AstraWeaveIpcServer",
    "serialize_value",
    "deserialize_request",
    "serialize_request",
    "serialize_response",
]


class IpcProtocolError(ApiError):
    """Raised for wire-format validation problems."""


@dataclass(frozen=True, slots=True)
class IpcRequestEnvelope:
    """Compatibility request envelope wrapper for older callers."""

    id: str
    method: str
    params: Mapping[str, Any] = field(default_factory=dict)
    caller: CallerIdentity | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.params, Mapping):
            raise IpcProtocolError(ApiErrorCode.INVALID_ARGUMENT, "params must be a mapping")
        _to_protocol_request(self.id, self.method, self.params, self.caller)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "IpcRequestEnvelope":
        protocol_request = validate_request_payload(payload)
        if not isinstance(protocol_request.params, Mapping):
            raise IpcProtocolError(ApiErrorCode.INVALID_ARGUMENT, "params must be a mapping")
        return cls(
            id=protocol_request.id,
            method=protocol_request.method,
            params=dict(protocol_request.params),
            caller=_caller_identity_from_envelope(protocol_request.caller),
        )

    def to_protocol(self) -> RequestEnvelope:
        return _to_protocol_request(self.id, self.method, self.params, self.caller)

    def to_dict(self) -> dict[str, Any]:
        return self.to_protocol().to_dict()


@dataclass(frozen=True, slots=True)
class IpcErrorEnvelope:
    """Compatibility error envelope wrapper."""

    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return ErrorPayload(code=self.code, message=self.message).to_dict()


@dataclass(frozen=True, slots=True)
class IpcSuccessEnvelope:
    """Compatibility success envelope wrapper."""

    id: str
    result: Any = None

    def to_dict(self) -> dict[str, Any]:
        return success_response(self.result, request_id=self.id).to_dict()


@dataclass(frozen=True, slots=True)
class IpcResponseEnvelope:
    """Compatibility union-like response envelope helper."""

    id: str
    ok: bool
    result: Any = None
    error: IpcErrorEnvelope | None = None

    def to_dict(self) -> dict[str, Any]:
        if self.ok:
            return success_response(self.result, request_id=self.id).to_dict()
        if self.error is None:
            err = ApiError(ApiErrorCode.INTERNAL, "internal server error")
            return error_response(err, request_id=self.id).to_dict()
        try:
            code = ApiErrorCode(self.error.code)
        except ValueError as exc:
            raise IpcProtocolError(ApiErrorCode.INVALID_ARGUMENT, "error code is not a known ApiErrorCode") from exc
        err = ApiError(code, self.error.message)
        return error_response(err, request_id=self.id).to_dict()


def parse_caller(payload: Any) -> CallerIdentity:
    """Parse a caller payload into a :class:`CallerIdentity`."""

    if isinstance(payload, CallerIdentity):
        return payload
    if isinstance(payload, CallerEnvelope):
        return CallerIdentity(user_sid=payload.user_sid, pid=payload.pid)
    if not isinstance(payload, Mapping):
        raise IpcProtocolError(ApiErrorCode.INVALID_ARGUMENT, "caller must be a mapping")
    try:
        parsed = CallerEnvelope.from_dict(payload)
    except ApiError as exc:
        raise IpcProtocolError(exc.code, exc.message) from exc
    return CallerIdentity(user_sid=parsed.user_sid, pid=parsed.pid)


def serialize_value(value: Any) -> Any:
    """Convert a Python object into a JSON-serializable representation."""

    return to_json_value(value)


def serialize_request(request: IpcRequestEnvelope | RequestEnvelope) -> dict[str, Any]:
    """Serialize a request envelope to a plain mapping."""

    if isinstance(request, IpcRequestEnvelope):
        return request.to_dict()
    if isinstance(request, RequestEnvelope):
        return request.to_dict()
    raise IpcProtocolError(ApiErrorCode.INVALID_ARGUMENT, "request must be an IpcRequestEnvelope or RequestEnvelope")


def deserialize_request(payload: Mapping[str, Any]) -> IpcRequestEnvelope:
    """Deserialize and validate a request envelope."""

    return IpcRequestEnvelope.from_mapping(payload)


def serialize_response(
    response: IpcResponseEnvelope | IpcSuccessEnvelope | SuccessResponse | ErrorResponse,
) -> dict[str, Any]:
    """Serialize a response envelope to a plain mapping."""

    if isinstance(response, (IpcResponseEnvelope, IpcSuccessEnvelope, SuccessResponse, ErrorResponse)):
        return response.to_dict()
    raise IpcProtocolError(ApiErrorCode.INVALID_ARGUMENT, "unsupported response envelope type")


def _error_from_exception(exc: BaseException) -> ApiError:
    if isinstance(exc, ApiError):
        return exc
    return ApiError(ApiErrorCode.INTERNAL, "internal server error")


def _caller_identity_from_envelope(caller: CallerEnvelope | None) -> CallerIdentity | None:
    if caller is None:
        return None
    return CallerIdentity(user_sid=caller.user_sid, pid=caller.pid)


def _to_protocol_request(
    request_id: str,
    method: str,
    params: Mapping[str, Any],
    caller: CallerIdentity | None,
) -> RequestEnvelope:
    caller_envelope = None
    if caller is not None:
        caller_envelope = CallerEnvelope(user_sid=caller.user_sid, pid=caller.pid)
    try:
        return RequestEnvelope(
            id=request_id,
            method=method,
            params=dict(params),
            caller=caller_envelope,
        )
    except ApiError as exc:
        raise IpcProtocolError(exc.code, exc.message) from exc


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


class AstraWeaveIpcServer:
    """Local IPC server that dispatches requests onto :class:`AstraWeaveService`."""

    def __init__(
        self,
        service: AstraWeaveService | None = None,
        *,
        prefer_named_pipe: bool = True,
        pipe_name: str | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
        authkey: bytes | None = None,
        require_explicit_caller: bool = True,
        bind_caller_per_connection: bool = True,
        enforce_runtime_caller_attestation: bool = True,
        caller_attestor: Callable[[CallerIdentity], SecurityDecision] | None = None,
    ) -> None:
        self._service = service or AstraWeaveService()
        self._prefer_named_pipe = prefer_named_pipe
        self._pipe_name = pipe_name or r"\\.\pipe\astrawave"
        self._host = host
        self._port = port
        self._authkey = authkey if authkey is not None else _authkey_from_env()
        self._require_explicit_caller = require_explicit_caller
        self._bind_caller_per_connection = bind_caller_per_connection
        self._enforce_runtime_caller_attestation = enforce_runtime_caller_attestation
        self._caller_attestor = caller_attestor or attest_caller_identity
        self._listener: Listener | None = None
        self._transport: str | None = None
        self._endpoint: Any = None
        self._stop_event = Event()
        self._lock = RLock()
        self._thread: Thread | None = None
        self._served_connections = 0
        self._served_requests = 0

    @property
    def service(self) -> AstraWeaveService:
        """Expose the wrapped service for tests and diagnostics."""

        return self._service

    @property
    def transport(self) -> str | None:
        """Return the active transport name if the server is started."""

        return self._transport

    @property
    def endpoint(self) -> Any:
        """Return the listener endpoint after startup."""

        return self._endpoint

    @property
    def authkey(self) -> bytes | None:
        """Return the configured listener authkey."""

        return self._authkey

    @property
    def is_running(self) -> bool:
        """Return whether the server loop is active."""

        return self._thread is not None and self._thread.is_alive()

    def __enter__(self) -> "AstraWeaveIpcServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def start(self) -> "AstraWeaveIpcServer":
        """Start the IPC server on a background thread."""

        with self._lock:
            if self.is_running:
                return self
            self._stop_event.clear()
            self._ensure_listener()
            self._thread = Thread(target=self.serve_forever, name="AstraWeaveIpcServer", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        """Stop the server and close the listener."""

        with self._lock:
            self._stop_event.set()
            if self._listener is not None:
                try:
                    self._listener.close()
                finally:
                    self._listener = None

            thread = self._thread
            self._thread = None

        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

    def serve_once(self) -> int:
        """Accept and handle exactly one connection."""

        if self._stop_event.is_set():
            return 0

        listener = self._ensure_listener()
        try:
            connection = listener.accept()
        except (OSError, EOFError):
            if self._stop_event.is_set():
                return 0
            raise

        self._served_connections += 1
        try:
            return self._serve_connection(connection)
        finally:
            connection.close()

    def serve_forever(self) -> int:
        """Accept connections until :meth:`stop` is called."""

        processed = 0
        while not self._stop_event.is_set():
            try:
                processed += self.serve_once()
            except (OSError, EOFError):
                if self._stop_event.is_set():
                    break
                raise
        return processed

    def handle_request(
        self,
        request: IpcRequestEnvelope | RequestEnvelope | Mapping[str, Any],
        *,
        expected_caller: CallerIdentity | None = None,
    ) -> dict[str, Any]:
        """Dispatch one request envelope and return a response mapping."""

        response_payload, _resolved_caller = self._process_request(request, expected_caller=expected_caller)
        return response_payload

    def _ensure_listener(self) -> Listener:
        with self._lock:
            if self._listener is not None:
                return self._listener

            listener, transport, endpoint = self._create_listener()
            self._listener = listener
            self._transport = transport
            self._endpoint = endpoint
            return listener

    def _create_listener(self) -> tuple[Listener, str, Any]:
        if self._prefer_named_pipe and os.name == "nt":
            try:
                listener = Listener(self._pipe_name, family="AF_PIPE", authkey=self._authkey)
                endpoint = self._pipe_name
                return listener, "named_pipe", endpoint
            except Exception:
                import logging
                logging.getLogger(__name__).warning(
                    "Named pipe creation failed, falling back to TCP"  # H20 fix
                )

        if not _is_loopback_host(self._host):
            raise ApiError(
                ApiErrorCode.INVALID_ARGUMENT,
                "IPC server host must be localhost or loopback in v1",
            )
        listener = Listener((self._host, self._port), family="AF_INET", authkey=self._authkey)
        endpoint = getattr(listener, "address", None) or getattr(listener, "_address", None)
        return listener, "localhost", endpoint

    def _serve_connection(self, connection) -> int:
        processed = 0
        bound_caller: CallerIdentity | None = None

        while not self._stop_event.is_set():
            try:
                payload = connection.recv()
            except EOFError:
                break

            expected = bound_caller if self._bind_caller_per_connection else None
            response, resolved_caller = self._process_request(payload, expected_caller=expected)
            if bound_caller is None and resolved_caller is not None and self._bind_caller_per_connection:
                bound_caller = resolved_caller

            try:
                connection.send(response)
            except (BrokenPipeError, OSError):
                break
            processed += 1
            self._served_requests += 1
        return processed

    def _process_request(
        self,
        request: IpcRequestEnvelope | RequestEnvelope | Mapping[str, Any],
        *,
        expected_caller: CallerIdentity | None,
    ) -> tuple[dict[str, Any], CallerIdentity | None]:
        request_id = self._extract_request_id(request)
        try:
            envelope = self._coerce_request(request)
            request_id = envelope.id
            caller = self._resolve_and_authorize_caller(envelope, expected_caller=expected_caller)
            result = self._dispatch(envelope, caller)
            return success_response(result, request_id=envelope.id).to_dict(), caller
        except BaseException as exc:
            err = _error_from_exception(exc)
            self._record_security_deny(error=err, request=request)
            return error_response(err, request_id=request_id or "invalid-request").to_dict(), None

    @staticmethod
    def _extract_request_id(request: IpcRequestEnvelope | RequestEnvelope | Mapping[str, Any]) -> str:
        request_id = getattr(request, "id", "")
        if isinstance(request_id, str) and request_id.strip():
            return request_id
        if isinstance(request, Mapping):
            raw = request.get("id", "")
            if isinstance(raw, str):
                return raw
            return str(raw) if raw is not None else ""
        return ""

    @staticmethod
    def _coerce_request(
        request: IpcRequestEnvelope | RequestEnvelope | Mapping[str, Any],
    ) -> RequestEnvelope:
        if isinstance(request, RequestEnvelope):
            AstraWeaveIpcServer._enforce_payload_size(request.to_dict())
            return request
        if isinstance(request, IpcRequestEnvelope):
            protocol_request = request.to_protocol()
            AstraWeaveIpcServer._enforce_payload_size(protocol_request.to_dict())
            return protocol_request
        if isinstance(request, Mapping):
            AstraWeaveIpcServer._enforce_payload_size(request)
            return validate_request_payload(request)
        raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "request must be a request envelope or mapping")

    def _resolve_and_authorize_caller(
        self,
        request: RequestEnvelope,
        *,
        expected_caller: CallerIdentity | None,
    ) -> CallerIdentity | None:
        caller = _caller_identity_from_envelope(request.caller)
        if self._require_explicit_caller and caller is None:
            raise ApiError(ApiErrorCode.AUTH_DENIED, "caller identity is required at IPC boundary")
        if caller is None:
            return None
        if expected_caller is not None and caller != expected_caller:
            raise ApiError(ApiErrorCode.AUTH_DENIED, "caller identity changed within the same IPC connection")
        if self._enforce_runtime_caller_attestation:
            attestation = self._caller_attestor(caller)
            if not attestation.allowed:
                raise attestation.to_api_error()
        admission = self._service.security_guard.authorize_caller(caller)
        if not admission.allowed:
            raise admission.to_api_error()
        return caller

    @staticmethod
    def _enforce_payload_size(payload: Mapping[str, Any]) -> None:
        size_bytes = estimate_json_size_bytes(payload)
        if size_bytes > MAX_CONTROL_PLANE_PAYLOAD_BYTES:
            raise ApiError(
                ApiErrorCode.INVALID_ARGUMENT,
                f"request payload exceeds maximum size of {MAX_CONTROL_PLANE_PAYLOAD_BYTES} bytes",
            )

    def _dispatch(self, request: RequestEnvelope, caller: CallerIdentity | None) -> Any:
        if not isinstance(request.params, Mapping):
            raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "params must be a mapping")
        params = dict(request.params)
        method = request.method

        if method == "CreateSession":
            return self._service.CreateSession(caller_identity=caller)
        if method == "LoadModel":
            session_id = self._require_str(params, "session_id")
            model_name = self._require_str(params, "model_name")
            runtime_backend = self._optional_str(params, "runtime_backend")
            runtime_profile = self._optional_str(params, "runtime_profile")
            runtime_backend_options = self._optional_tuning_options(params, "runtime_backend_options")
            self._call_service_method(
                "LoadModel",
                session_id,
                model_name,
                caller_identity=caller,
                runtime_backend=runtime_backend,
                runtime_profile=runtime_profile,
                backend_options=runtime_backend_options,
                runtime_backend_options=runtime_backend_options,
            )
            return None
        if method == "RegisterTensor":
            session_id = self._require_str(params, "session_id")
            tensor_name = self._require_str(params, "tensor_name")
            size_bytes = self._require_int(params, "size_bytes")
            self._service.RegisterTensor(session_id, tensor_name, size_bytes, caller_identity=caller)
            return None
        if method == "SetTierHint":
            session_id = self._require_str(params, "session_id")
            tensor_name = self._require_str(params, "tensor_name")
            tier = self._parse_memory_tier(params.get("tier"))
            self._service.SetTierHint(session_id, tensor_name, tier, caller_identity=caller)
            return None
        if method == "PrefetchPlan":
            session_id = self._require_str(params, "session_id")
            return self._service.PrefetchPlan(session_id, caller_identity=caller)
        if method == "RunStep":
            session_id = self._require_str(params, "session_id")
            step_name = params.get("step_name", "run")
            if not isinstance(step_name, str) or not step_name.strip():
                raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "step_name must be a non-empty string")
            prompt = self._optional_str(params, "prompt")
            max_tokens = self._optional_int(params, "max_tokens")
            temperature = self._optional_float(params, "temperature")
            runtime_profile_override = self._optional_str(params, "runtime_profile_override")
            runtime_backend_options_override = self._optional_tuning_options(
                params,
                "runtime_backend_options_override",
            )
            return self._call_service_method(
                "RunStep",
                session_id,
                step_name=step_name,
                caller_identity=caller,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                runtime_profile_override=runtime_profile_override,
                backend_options=runtime_backend_options_override,
                runtime_backend_options_override=runtime_backend_options_override,
            )
        if method == "GetResidency":
            session_id = self._require_str(params, "session_id")
            return self._service.GetResidency(session_id, caller_identity=caller)
        if method == "GetPressure":
            session_id = self._require_str(params, "session_id")
            return self._service.GetPressure(session_id, caller_identity=caller)
        if method == "SetPolicy":
            session_id = self._require_str(params, "session_id")
            policy = self._parse_policy_profile(params.get("policy"))
            self._service.SetPolicy(session_id, policy, caller_identity=caller)
            return None
        if method == "CloseSession":
            session_id = self._require_str(params, "session_id")
            self._service.CloseSession(session_id, caller_identity=caller)
            return None

        raise ApiError(ApiErrorCode.INVALID_ARGUMENT, f"unknown IPC method: {method}")

    @staticmethod
    def _require_str(params: Mapping[str, Any], key: str) -> str:
        value = params.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ApiError(ApiErrorCode.INVALID_ARGUMENT, f"{key} must be a non-empty string")
        return value

    @staticmethod
    def _require_int(params: Mapping[str, Any], key: str) -> int:
        value = params.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ApiError(ApiErrorCode.INVALID_ARGUMENT, f"{key} must be a positive integer")
        return value

    @staticmethod
    def _optional_str(params: Mapping[str, Any], key: str) -> str | None:
        value = params.get(key)
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ApiError(ApiErrorCode.INVALID_ARGUMENT, f"{key} must be a non-empty string")
        return value

    @staticmethod
    def _optional_int(params: Mapping[str, Any], key: str) -> int | None:
        value = params.get(key)
        if value is None:
            return None
        if not isinstance(value, int) or isinstance(value, bool):
            raise ApiError(ApiErrorCode.INVALID_ARGUMENT, f"{key} must be an integer")
        if value <= 0:
            raise ApiError(ApiErrorCode.INVALID_ARGUMENT, f"{key} must be a positive integer")
        return value

    @staticmethod
    def _optional_float(params: Mapping[str, Any], key: str) -> float | None:
        value = params.get(key)
        if value is None:
            return None
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ApiError(ApiErrorCode.INVALID_ARGUMENT, f"{key} must be a number")
        return float(value)

    @staticmethod
    def _optional_tuning_options(params: Mapping[str, Any], key: str) -> dict[str, Any] | None:
        value = params.get(key)
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise ApiError(ApiErrorCode.INVALID_ARGUMENT, f"{key} must be a JSON object")

        normalized: dict[str, Any] = {}
        for option_key, option_value in value.items():
            if not isinstance(option_key, str) or not option_key.strip():
                raise ApiError(ApiErrorCode.INVALID_ARGUMENT, f"{key} keys must be non-empty strings")
            normalized[option_key.strip()] = option_value
        return normalized

    def _call_service_method(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        method = getattr(self._service, method_name)
        try:
            params = signature(method).parameters
        except (TypeError, ValueError):  # pragma: no cover - unusual callable
            return method(*args, **kwargs)

        if any(param.kind == Parameter.VAR_KEYWORD for param in params.values()):
            return method(*args, **kwargs)

        filtered_kwargs = {key: value for key, value in kwargs.items() if key in params}
        return method(*args, **filtered_kwargs)

    @staticmethod
    def _parse_memory_tier(value: Any) -> MemoryTier:
        if isinstance(value, MemoryTier):
            return value
        if isinstance(value, str):
            try:
                return MemoryTier(value)
            except ValueError as exc:
                raise ApiError(ApiErrorCode.INVALID_ARGUMENT, f"unknown MemoryTier: {value}") from exc
        raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "tier must be a MemoryTier value")

    @staticmethod
    def _parse_policy_profile(value: Any) -> PolicyProfile:
        if isinstance(value, PolicyProfile):
            return value
        if isinstance(value, str):
            try:
                return PolicyProfile(value)
            except ValueError as exc:
                raise ApiError(ApiErrorCode.INVALID_ARGUMENT, f"unknown PolicyProfile: {value}") from exc
        raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "policy must be a PolicyProfile value")

    def _record_security_deny(
        self,
        *,
        error: ApiError,
        request: IpcRequestEnvelope | RequestEnvelope | Mapping[str, Any],
    ) -> None:
        if error.code not in {ApiErrorCode.AUTH_DENIED, ApiErrorCode.RATE_LIMITED}:
            return
        request_id = self._extract_request_id(request) or "unknown"
        caller_id = "unknown"
        if isinstance(request, RequestEnvelope) and request.caller is not None:
            caller_id = f"{request.caller.user_sid}:{request.caller.pid}"
        elif isinstance(request, IpcRequestEnvelope) and request.caller is not None:
            caller_id = f"{request.caller.user_sid}:{request.caller.pid}"
        elif isinstance(request, Mapping):
            caller_raw = request.get("caller")
            if isinstance(caller_raw, Mapping):
                sid = caller_raw.get("user_sid")
                pid = caller_raw.get("pid")
                if isinstance(sid, str) and isinstance(pid, int):
                    caller_id = f"{sid}:{pid}"
        try:
            self._service.telemetry.record_event(
                SecurityEvent(
                    reason_code=TelemetryReasonCode.SECURITY_DENY,
                    session_id=request_id,
                    caller_id=caller_id,
                    endpoint=str(self._endpoint),
                    decision=error.code.value,
                ),
                extra_identifiers={"request_id": request_id},
            )
        except Exception:  # H21 fix: log telemetry recording failures
            import logging
            logging.getLogger(__name__).debug("Failed to record security telemetry event", exc_info=True)
            return
