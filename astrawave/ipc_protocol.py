"""JSON-serializable IPC envelopes for AstraWeave.

This module stays dependency-light on purpose. It defines the strict request
and response shapes used by the IPC boundary, plus deterministic helpers for
validation, serialization, and API error mapping.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from math import isfinite
from json import dumps, loads
from typing import Any, Mapping, MutableMapping

from .errors import ApiError, ApiErrorCode


JsonPrimitive = str | int | float | bool | None
JsonValue = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
MAX_CONTROL_PLANE_PAYLOAD_BYTES = 1 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class CallerEnvelope:
    """JSON caller identity for IPC messages."""

    user_sid: str
    pid: int

    def __post_init__(self) -> None:
        _validate_caller_fields(self.user_sid, self.pid)

    def to_dict(self) -> dict[str, JsonValue]:
        return {"user_sid": self.user_sid, "pid": self.pid}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CallerEnvelope":
        user_sid = _require_string(data, "user_sid")
        pid = _require_int(data, "pid")
        return cls(user_sid=user_sid, pid=pid)


@dataclass(frozen=True, slots=True)
class RequestEnvelope:
    """Strict IPC request envelope."""

    id: str
    method: str
    params: Any = None
    caller: CallerEnvelope | None = None

    def __post_init__(self) -> None:
        _validate_request_fields(self.id, self.method, self.params, self.caller)

    def to_dict(self) -> dict[str, JsonValue]:
        _validate_exact_keys(self, {"id", "method", "params", "caller"})
        data: dict[str, JsonValue] = {
            "id": self.id,
            "method": self.method,
            "params": to_json_value(self.params),
            "caller": self.caller.to_dict() if self.caller is not None else None,
        }
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RequestEnvelope":
        _validate_exact_mapping_keys(data, {"id", "method", "params", "caller"})
        request_id = _require_string(data, "id")
        method = _require_string(data, "method")
        params = data["params"]
        caller_raw = data.get("caller")
        caller = None
        if caller_raw is not None:
            if not isinstance(caller_raw, Mapping):
                raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "caller must be an object or null")
            caller = CallerEnvelope.from_dict(caller_raw)
        return cls(id=request_id, method=method, params=params, caller=caller)


@dataclass(frozen=True, slots=True)
class ErrorPayload:
    """Structured IPC error payload."""

    code: str
    message: str

    def __post_init__(self) -> None:
        if not self.code or not self.code.strip():
            raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "error code must be non-empty")
        if not self.message or not self.message.strip():
            raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "error message must be non-empty")

    def to_dict(self) -> dict[str, JsonValue]:
        _validate_exact_keys(self, {"code", "message"})
        return {"code": self.code, "message": self.message}

    @classmethod
    def from_api_error(cls, error: ApiError) -> "ErrorPayload":
        return cls(code=error.code.value, message=error.message)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ErrorPayload":
        code = _require_string(data, "code")
        message = _require_string(data, "message")
        return cls(code=code, message=message)

    def to_api_error(self) -> ApiError:
        return ApiError(api_error_code_from_string(self.code), self.message)


@dataclass(frozen=True, slots=True)
class SuccessResponse:
    """Success response envelope."""

    id: str
    ok: bool = field(default=True, init=False)
    result: Any = None

    def __post_init__(self) -> None:
        if not self.id or not self.id.strip():
            raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "response id must be non-empty")

    def to_dict(self) -> dict[str, JsonValue]:
        _validate_exact_keys(self, {"id", "ok", "result"})
        return {
            "id": self.id,
            "ok": True,
            "result": to_json_value(self.result),
        }


@dataclass(frozen=True, slots=True)
class ErrorResponse:
    """Error response envelope."""

    id: str
    ok: bool = field(default=False, init=False)
    error: ErrorPayload = field(default_factory=lambda: ErrorPayload(ApiErrorCode.INTERNAL.value, "internal error"))

    def __post_init__(self) -> None:
        if not self.id or not self.id.strip():
            raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "response id must be non-empty")

    def to_dict(self) -> dict[str, JsonValue]:
        _validate_exact_keys(self, {"id", "ok", "error"})
        return {
            "id": self.id,
            "ok": False,
            "error": self.error.to_dict(),
        }


def to_json_value(value: Any) -> JsonValue:
    """Validate and return a JSON-serializable value.

    The function accepts native JSON types, lists/tuples, and mappings with
    string keys. Anything else is rejected with a stable `ApiError`.
    """

    if isinstance(value, Enum):
        return to_json_value(value.value)
    if is_dataclass(value):
        return to_json_value(asdict(value))
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not isfinite(value):
            raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "JSON numbers must be finite")
        return value
    if isinstance(value, tuple):
        return [to_json_value(item) for item in value]
    if isinstance(value, list):
        return [to_json_value(item) for item in value]
    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "JSON object keys must be strings")
            result[key] = to_json_value(item)
        return result
    raise ApiError(ApiErrorCode.INVALID_ARGUMENT, f"value is not JSON serializable: {type(value).__name__}")


def validate_request_payload(payload: Mapping[str, Any]) -> RequestEnvelope:
    """Validate and normalize a raw request payload."""

    _validate_request_payload_size(payload)
    return RequestEnvelope.from_dict(payload)


def request_to_json(request: RequestEnvelope, *, indent: int | None = None) -> str:
    """Serialize a validated request envelope to JSON."""

    return dumps(request.to_dict(), ensure_ascii=True, indent=indent)


def request_from_json(payload: str) -> RequestEnvelope:
    """Deserialize a request envelope from JSON."""

    data = loads(payload)
    if not isinstance(data, MutableMapping):
        raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "request JSON must decode to an object")
    return validate_request_payload(data)


def success_response(result: Any, *, request_id: str) -> SuccessResponse:
    """Build a success response envelope."""

    return SuccessResponse(id=request_id, result=result)


def error_response(error: ApiError, *, request_id: str) -> ErrorResponse:
    """Build an error response envelope from a typed API error."""

    return ErrorResponse(id=request_id, error=ErrorPayload.from_api_error(error))


def response_to_json(response: SuccessResponse | ErrorResponse, *, indent: int | None = None) -> str:
    """Serialize a response envelope to JSON."""

    return dumps(response.to_dict(), ensure_ascii=True, indent=indent)


def response_from_json(payload: str) -> SuccessResponse | ErrorResponse:
    """Deserialize a response envelope from JSON."""

    data = loads(payload)
    if not isinstance(data, MutableMapping):
        raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "response JSON must decode to an object")
    return validate_response_payload(data)


def validate_response_payload(payload: Mapping[str, Any]) -> SuccessResponse | ErrorResponse:
    """Validate and normalize a raw response payload."""

    response_id = _require_string(payload, "id")
    ok = _require_bool(payload, "ok")
    if ok:
        _validate_exact_mapping_keys(payload, {"id", "ok", "result"})
        return SuccessResponse(id=response_id, result=payload.get("result"))
    _validate_exact_mapping_keys(payload, {"id", "ok", "error"})
    error_raw = payload.get("error")
    if not isinstance(error_raw, Mapping):
        raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "error response must include an error object")
    return ErrorResponse(id=response_id, error=ErrorPayload.from_dict(error_raw))


def api_error_code_from_string(value: str) -> ApiErrorCode:
    """Resolve a string code to a stable `ApiErrorCode`."""

    try:
        return ApiErrorCode(value)
    except ValueError as exc:
        raise ApiError(ApiErrorCode.INVALID_ARGUMENT, f"unknown API error code: {value}") from exc


def api_error_from_payload(payload: Mapping[str, Any]) -> ApiError:
    """Build a typed `ApiError` from an IPC error payload."""

    return ErrorPayload.from_dict(payload).to_api_error()


def api_error_to_payload(error: ApiError) -> ErrorPayload:
    """Convert a typed `ApiError` to an IPC error payload."""

    return ErrorPayload.from_api_error(error)


def estimate_json_size_bytes(value: Any) -> int:
    """Estimate UTF-8 JSON payload size without full serialization.

    Uses ``sys.getsizeof`` on the ``str()`` representation as a cheap
    upper-bound estimator, avoiding a full ``json.dumps`` round-trip on
    every request validation call.
    """

    return sys.getsizeof(str(value))


def _validate_request_fields(
    request_id: str,
    method: str,
    params: Any,
    caller: CallerEnvelope | None,
) -> None:
    if not request_id or not request_id.strip():
        raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "request id must be non-empty")
    if not method or not method.strip():
        raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "method must be non-empty")
    to_json_value(params)
    if caller is not None and not isinstance(caller, CallerEnvelope):
        raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "caller must be a CallerEnvelope or null")


def _validate_caller_fields(user_sid: str, pid: int) -> None:
    if not user_sid or not user_sid.strip():
        raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "user_sid must be non-empty")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "pid must be a positive integer")


def _require_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ApiError(ApiErrorCode.INVALID_ARGUMENT, f"{key} must be a non-empty string")
    return value


def _require_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ApiError(ApiErrorCode.INVALID_ARGUMENT, f"{key} must be a positive integer")
    return value


def _require_bool(payload: Mapping[str, Any], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ApiError(ApiErrorCode.INVALID_ARGUMENT, f"{key} must be a boolean")
    return value


def _validate_exact_mapping_keys(payload: Mapping[str, Any], allowed_keys: set[str]) -> None:
    payload_keys = set(payload.keys())
    extra_keys = payload_keys - allowed_keys
    missing_keys = allowed_keys - payload_keys
    if extra_keys or missing_keys:
        message_parts: list[str] = []
        if missing_keys:
            message_parts.append(f"missing keys: {sorted(missing_keys)}")
        if extra_keys:
            message_parts.append(f"unexpected keys: {sorted(extra_keys)}")
        raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "; ".join(message_parts))


def _validate_exact_keys(obj: Any, allowed_keys: set[str]) -> None:
    if not hasattr(obj, "__dict__") and not hasattr(obj, "__slots__"):
        return
    object_keys = {
        field_name
        for field_name in getattr(obj, "__dataclass_fields__", {})
    }
    extra_keys = object_keys - allowed_keys
    missing_keys = allowed_keys - object_keys
    if extra_keys or missing_keys:
        message_parts: list[str] = []
        if missing_keys:
            message_parts.append(f"missing fields: {sorted(missing_keys)}")
        if extra_keys:
            message_parts.append(f"unexpected fields: {sorted(extra_keys)}")
        raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "; ".join(message_parts))


def _validate_request_payload_size(payload: Mapping[str, Any]) -> None:
    size_bytes = estimate_json_size_bytes(payload)
    if size_bytes > MAX_CONTROL_PLANE_PAYLOAD_BYTES:
        raise ApiError(
            ApiErrorCode.INVALID_ARGUMENT,
            f"request payload exceeds maximum size of {MAX_CONTROL_PLANE_PAYLOAD_BYTES} bytes",
        )


__all__ = [
    "ApiError",
    "ApiErrorCode",
    "CallerEnvelope",
    "ErrorPayload",
    "ErrorResponse",
    "JsonValue",
    "RequestEnvelope",
    "SuccessResponse",
    "MAX_CONTROL_PLANE_PAYLOAD_BYTES",
    "api_error_code_from_string",
    "api_error_from_payload",
    "api_error_to_payload",
    "estimate_json_size_bytes",
    "error_response",
    "request_from_json",
    "request_to_json",
    "response_from_json",
    "response_to_json",
    "success_response",
    "to_json_value",
    "validate_request_payload",
    "validate_response_payload",
]
