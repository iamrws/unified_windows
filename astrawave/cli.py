"""AstraWeave CLI."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from contextlib import suppress
import getpass
import hashlib
import hmac
from ipaddress import ip_address
import json
import os
import sys
import tempfile
from pathlib import Path
from time import monotonic, sleep, time_ns
from typing import Any, Mapping, Protocol, Sequence
from uuid import uuid4

from .errors import ApiError, ApiErrorCode
from .fallback import FallbackController, FallbackState, FallbackStep
from .security import CallerIdentity
from .types import MemoryTier, PolicyProfile, PressureSnapshot, ResidencySnapshot, ResidencyState, SessionState


class _Backend(Protocol):
    """Protocol shared by LocalBackend and RemoteBackend for _dispatch()."""

    def create_session(self, caller: CallerIdentity) -> dict[str, Any]: ...
    def load_model(
        self,
        session_id: str,
        model_name: str,
        caller: CallerIdentity,
        runtime_backend: str | None = ...,
        runtime_profile: str | None = ...,
        runtime_backend_options: dict[str, Any] | None = ...,
    ) -> dict[str, Any]: ...
    def register_tensor(self, session_id: str, tensor_name: str, size_bytes: int, caller: CallerIdentity) -> dict[str, Any]: ...
    def set_tier_hint(self, session_id: str, tensor_name: str, tier: MemoryTier, caller: CallerIdentity) -> dict[str, Any]: ...
    def prefetch_plan(self, session_id: str, caller: CallerIdentity) -> Any: ...
    def run_step(
        self,
        session_id: str,
        step_name: str,
        caller: CallerIdentity,
        prompt: str | None = ...,
        max_tokens: int | None = ...,
        temperature: float | None = ...,
        runtime_profile_override: str | None = ...,
        runtime_backend_options_override: dict[str, Any] | None = ...,
    ) -> Any: ...
    def get_residency(self, session_id: str, caller: CallerIdentity) -> Any: ...
    def get_pressure(self, session_id: str, caller: CallerIdentity) -> Any: ...
    def set_policy(self, session_id: str, policy: PolicyProfile, caller: CallerIdentity) -> dict[str, Any]: ...
    def close_session(self, session_id: str, caller: CallerIdentity) -> dict[str, Any]: ...


try:
    from .sdk import AstraWeaveSDK
except ImportError:  # pragma: no cover - optional transport stack
    AstraWeaveSDK = None

try:
    from .ipc_server import AstraWeaveIpcServer
except ImportError:  # pragma: no cover - optional phase-3 transport stack
    AstraWeaveIpcServer = None

try:
    from .service_host import AstraWeaveServiceHost, ServiceHostConfig
except ImportError:  # pragma: no cover - optional phase-3 host wrapper
    AstraWeaveServiceHost = None
    ServiceHostConfig = None

try:
    from .hardware_probe import collect_hardware_probe as _collect_hardware_probe_impl
except ImportError:  # pragma: no cover - optional hardware probe module
    _collect_hardware_probe_impl = None

try:
    from .service import DEFAULT_SERVICE_OWNER_SID
except ImportError:  # pragma: no cover - future-proof fallback
    DEFAULT_SERVICE_OWNER_SID = "S-1-5-21-AstraWeave-Owner"

STATE_DIR = Path(tempfile.gettempdir()) / "astrawave_cli_state"
STATE_VERSION = 1
DEFAULT_ENDPOINT = "auto"
HIGH_PRESSURE_THRESHOLD = 0.75
DEFAULT_SERVE_ENDPOINT = "auto"
DEFAULT_SERVE_DURATION_SECONDS: float | None = None
_POSITIVE_INT_OPTION_KEYS = {"num_ctx", "num_batch", "num_predict", "num_keep", "top_k"}
_NONNEGATIVE_INT_OPTION_KEYS = {"gpu_layers", "num_gpu"}
_NONNEGATIVE_FLOAT_OPTION_KEYS = {"temperature"}
_OPEN_INTERVAL_FLOAT_OPTION_KEYS = {"top_p", "repeat_penalty"}


def _derive_state_hmac_key() -> bytes:
    """Derive a per-user HMAC key from the username and state directory path."""
    user = getpass.getuser()
    material = f"{user}|{STATE_DIR}".encode("utf-8")
    return hashlib.sha256(material).digest()


def _compute_state_hmac(json_bytes: bytes) -> str:
    """Compute an HMAC-SHA256 hex digest for state file content."""
    return hmac.new(_derive_state_hmac_key(), json_bytes, hashlib.sha256).hexdigest()


def _now_ms() -> int:
    return time_ns() // 1_000_000


def _caller_key(caller: CallerIdentity) -> str:
    return f"{caller.user_sid}:{caller.pid}"


def _endpoint_path(endpoint: str) -> Path:
    digest = hashlib.sha256(endpoint.encode("utf-8")).hexdigest()
    return STATE_DIR / f"{digest}.json"


def _valid_caller(caller: CallerIdentity | None) -> bool:
    return bool(caller and caller.user_sid.strip() and caller.pid > 0)


def _to_memory_tier(value: str | MemoryTier) -> MemoryTier:
    return value if isinstance(value, MemoryTier) else MemoryTier(value)


def _to_policy_profile(value: str | PolicyProfile) -> PolicyProfile:
    return value if isinstance(value, PolicyProfile) else PolicyProfile(value)


def _residency_for_tier(tier: MemoryTier) -> ResidencyState:
    if tier is MemoryTier.HOT:
        return ResidencyState.VRAM
    if tier is MemoryTier.WARM:
        return ResidencyState.PINNED_RAM
    return ResidencyState.PAGEABLE_RAM


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (MemoryTier, PolicyProfile, ResidencyState, SessionState, FallbackStep)):
        return value.value
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_safe(item) for item in value]
    return value


def _write_json(payload: Mapping[str, Any], *, stream: str = "stdout") -> None:
    text = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    if stream == "stderr":
        print(text, file=sys.stderr)
    else:
        print(text)


def _emit_ok(result: Mapping[str, Any]) -> None:
    _write_json({"ok": True, "result": result})


def _emit_error(error: ApiError, *, stream: str = "stderr") -> None:
    _write_json({"ok": False, "error": {"code": error.code.value, "message": error.message}}, stream=stream)


def _default_state() -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "service_owner_sid": DEFAULT_SERVICE_OWNER_SID,
        "create_session_attempts": {},
        "active_sessions": {},
        "sessions": {},
    }


def _normalize_endpoint_hint(endpoint: str | None) -> str:
    value = (endpoint or "").strip()
    if value in {"", DEFAULT_ENDPOINT, "local://default"}:
        return DEFAULT_SERVE_ENDPOINT
    return value


def _describe_endpoint(endpoint: Any) -> str:
    if isinstance(endpoint, tuple) and len(endpoint) == 2:
        host, port = endpoint
        return f"tcp://{host}:{port}"
    if isinstance(endpoint, str):
        if endpoint.startswith("\\\\.\\pipe\\"):
            return f"pipe://{endpoint}"
        return endpoint
    return str(endpoint)


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
        raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "host must be localhost or loopback in v1")
    return normalized


def _parse_port_value(text: str, *, allow_zero: bool) -> int:
    try:
        port = int(text)
    except ValueError as exc:
        raise ApiError(ApiErrorCode.INVALID_ARGUMENT, f"invalid TCP port: {text}") from exc
    minimum = 0 if allow_zero else 1
    if port < minimum or port > 65535:
        bound = "0-65535" if allow_zero else "1-65535"
        raise ApiError(ApiErrorCode.INVALID_ARGUMENT, f"TCP port must be in range {bound}")
    return port


def _parse_runtime_option_pairs(
    raw_options: Sequence[str] | None,
    *,
    field_name: str,
) -> dict[str, Any] | None:
    if raw_options is None:
        return None
    parsed: dict[str, Any] = {}
    for raw_item in raw_options:
        if "=" not in raw_item:
            raise ApiError(ApiErrorCode.INVALID_ARGUMENT, f"{field_name} entries must use key=value syntax")
        key_text, raw_value = raw_item.split("=", 1)
        key = key_text.strip()
        if not key:
            raise ApiError(ApiErrorCode.INVALID_ARGUMENT, f"{field_name} keys must be non-empty strings")
        parsed[key] = _parse_runtime_option_value(field_name, key, raw_value.strip())
    return parsed


def _parse_runtime_option_value(field_name: str, key: str, raw_value: str) -> Any:
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value

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
    return value


def _normalize_runtime_profile(field_name: str, value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ApiError(ApiErrorCode.INVALID_ARGUMENT, f"{field_name} must be a non-empty string")
    return value.strip()


def _parse_serve_endpoint(endpoint: str, transport: str) -> dict[str, Any]:
    endpoint_hint = _normalize_endpoint_hint(endpoint)
    transport_hint = transport.strip().lower()
    if transport_hint not in {"auto", "tcp", "pipe"}:
        raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "transport must be one of: auto, tcp, pipe")

    if endpoint_hint.startswith("pipe://"):
        pipe_name = endpoint_hint.removeprefix("pipe://") or r"\\.\pipe\astrawave"
        return {
            "prefer_named_pipe": True,
            "pipe_name": pipe_name,
        }
    if endpoint_hint.startswith("\\\\.\\pipe\\"):
        return {
            "prefer_named_pipe": True,
            "pipe_name": endpoint_hint,
        }
    if endpoint_hint.startswith("tcp://"):
        host, port = _parse_tcp_endpoint(endpoint_hint.removeprefix("tcp://"))
        return {
            "prefer_named_pipe": False,
            "host": host,
            "port": port,
        }

    if transport_hint == "pipe":
        return {
            "prefer_named_pipe": True,
            "pipe_name": r"\\.\pipe\astrawave",
        }
    if transport_hint == "tcp":
        host, port = _parse_tcp_endpoint(endpoint_hint)
        return {
            "prefer_named_pipe": False,
            "host": host,
            "port": port,
        }

    if os.name == "nt":
        return {
            "prefer_named_pipe": True,
            "pipe_name": r"\\.\pipe\astrawave",
        }
    return {
        "prefer_named_pipe": False,
        "host": "127.0.0.1",
        "port": 0,
    }


def _parse_tcp_endpoint(value: str) -> tuple[str, int]:
    hint = value.strip()
    if not hint or hint == DEFAULT_SERVE_ENDPOINT:
        return "127.0.0.1", 0
    if hint == "auto":
        return "127.0.0.1", 0
    if ":" in hint:
        host, _, port_text = hint.rpartition(":")
        if not host:
            host = "127.0.0.1"
        host = _require_loopback_host(host)
        return host, _parse_port_value(port_text, allow_zero=True)
    return _require_loopback_host(hint), 0


def _hardware_probe_result() -> dict[str, Any]:
    if _collect_hardware_probe_impl is not None:
        try:
            probe = _collect_hardware_probe_impl()
        except Exception as exc:  # pragma: no cover - defensive fallback
            raise ApiError(ApiErrorCode.INTERNAL, "hardware probe failed unexpectedly") from exc
        if isinstance(probe, Mapping):
            return dict(probe)
        raise ApiError(ApiErrorCode.INTERNAL, "hardware probe returned an invalid payload")

    warning = "hardware probe module is unavailable in this build"
    return {
        "timestamp_ms": _now_ms(),
        "nvidia_smi": {
            "available": False,
            "status": "unavailable",
            "devices": [],
            "error": {
                "code": "HARDWARE_PROBE_MODULE_MISSING",
                "message": warning,
            },
        },
        "nvml": {
            "available": False,
            "status": "unavailable",
            "devices": [],
            "error": {
                "code": "HARDWARE_PROBE_MODULE_MISSING",
                "message": warning,
            },
        },
        "effective": {
            "source": "none",
            "device_count": 0,
            "driver_version": None,
            "devices": [],
            "has_nvidia_gpu": False,
        },
        "warnings": [warning],
    }


class LocalBackend:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self.path = _endpoint_path(endpoint)
        self.state = self._load()
        self.fallback = FallbackController()

    def create_session(self, caller: CallerIdentity) -> dict[str, Any]:
        self._authorize(caller)
        now_ms = _now_ms()
        self._record_attempt(caller, now_ms)
        self._enforce_rate_limit(caller, now_ms)
        self._enforce_session_cap(caller)
        session_id = str(uuid4())
        self.state["sessions"][session_id] = {
            "session_id": session_id,
            "owner_sid": caller.user_sid,
            "owner_pid": caller.pid,
        "state": SessionState.SESSION_CREATED.value,
        "model_name": None,
        "runtime_backend": None,
        "policy_profile": PolicyProfile.STABILITY.value,
        "tensors": {},
        "active_run": False,
            "closed": False,
            "fallback_current_step": None,
            "fallback_last_step_change_ms": None,
            "fallback_step_change_history_ms": [],
            "fallback_stability_mode": False,
            "vram_budget_bytes": 8 * 1024**3,
            "vram_used_bytes": 0,
            "pinned_ram_used_bytes": 0,
            "pageable_ram_used_bytes": 0,
            "cpu_only_used_bytes": 0,
        }
        self.state["active_sessions"][_caller_key(caller)] = self.state["active_sessions"].get(_caller_key(caller), 0) + 1
        self._save()
        return {"session_id": session_id, "owner_sid": caller.user_sid, "owner_pid": caller.pid}

    def load_model(
        self,
        session_id: str,
        model_name: str,
        caller: CallerIdentity,
        runtime_backend: str | None = None,
        runtime_profile: str | None = None,
        runtime_backend_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        session = self._get_owned_session(session_id, caller, "LoadModel")
        self._require_state(session, {SessionState.SESSION_CREATED}, "LoadModel")
        if not model_name:
            raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "model_name must not be empty")
        normalized_profile = _normalize_runtime_profile("runtime_profile", runtime_profile)
        session["model_name"] = model_name
        session["runtime_backend"] = runtime_backend
        session["runtime_profile"] = normalized_profile
        session["runtime_backend_options"] = runtime_backend_options or {}
        session["state"] = SessionState.READY.value
        self._save()
        result = {"session_id": session_id, "model_name": model_name, "state": session["state"]}
        if runtime_backend is not None:
            result["runtime_backend"] = runtime_backend
        if normalized_profile is not None:
            result["runtime_profile"] = normalized_profile
        if runtime_backend_options is not None:
            result["runtime_backend_options"] = runtime_backend_options
        return result

    def register_tensor(self, session_id: str, tensor_name: str, size_bytes: int, caller: CallerIdentity) -> dict[str, Any]:
        session = self._get_owned_session(session_id, caller, "RegisterTensor")
        self._require_state(session, {SessionState.MODEL_LOADED, SessionState.READY, SessionState.DEGRADED}, "RegisterTensor")
        if not tensor_name:
            raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "tensor_name must not be empty")
        if size_bytes <= 0:
            raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "size_bytes must be positive")
        tensors = session["tensors"]
        if tensor_name in tensors:
            raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "tensor already registered")
        tensors[tensor_name] = {"name": tensor_name, "size_bytes": size_bytes, "tier_hint": MemoryTier.WARM.value, "residency": ResidencyState.PINNED_RAM.value}
        session["pinned_ram_used_bytes"] += size_bytes
        session["state"] = SessionState.READY.value
        self._save()
        return {"session_id": session_id, "tensor_name": tensor_name, "size_bytes": size_bytes}

    def set_tier_hint(self, session_id: str, tensor_name: str, tier: MemoryTier, caller: CallerIdentity) -> dict[str, Any]:
        session = self._get_owned_session(session_id, caller, "SetTierHint")
        self._require_state(session, {SessionState.MODEL_LOADED, SessionState.READY, SessionState.DEGRADED}, "SetTierHint")
        tensor = self._get_tensor(session, tensor_name)
        tensor["tier_hint"] = _to_memory_tier(tier).value
        self._save()
        return {"session_id": session_id, "tensor_name": tensor_name, "tier_hint": tensor["tier_hint"]}

    def prefetch_plan(self, session_id: str, caller: CallerIdentity) -> dict[str, Any]:
        session = self._get_owned_session(session_id, caller, "PrefetchPlan")
        self._require_state(session, {SessionState.READY, SessionState.DEGRADED}, "PrefetchPlan")
        migrations = []
        for name in sorted(session["tensors"]):
            tensor = session["tensors"][name]
            destination = _residency_for_tier(MemoryTier(tensor["tier_hint"]))
            if tensor["residency"] != destination.value:
                migrations.append(self._migrate_tensor(session, tensor, destination, reason_code="prefetch"))
        self._save()
        return {"session_id": session_id, "migrations": migrations}

    def run_step(
        self,
        session_id: str,
        step_name: str,
        caller: CallerIdentity,
        prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        runtime_profile_override: str | None = None,
        runtime_backend_options_override: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        session = self._get_owned_session(session_id, caller, "RunStep")
        self._require_state(session, {SessionState.READY, SessionState.DEGRADED}, "RunStep")
        if not step_name:
            raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "step_name must not be empty")
        normalized_profile_override = _normalize_runtime_profile(
            "runtime_profile_override",
            runtime_profile_override,
        )
        if session["active_run"]:
            raise ApiError(ApiErrorCode.CONFLICT_RUN_IN_PROGRESS, "a RunStep is already active")
        session["active_run"] = True
        session["state"] = SessionState.RUNNING.value
        try:
            pressure = self._compute_pressure(session)
            fallback_result = None
            if pressure >= HIGH_PRESSURE_THRESHOLD or session["fallback_stability_mode"]:
                fallback_result = self._advance_fallback(session, caller)
            if session["state"] != SessionState.FAILED.value:
                session["state"] = SessionState.DEGRADED.value if (
                    session["fallback_stability_mode"] or session["fallback_current_step"] or pressure >= HIGH_PRESSURE_THRESHOLD
                ) else SessionState.READY.value
            result = {
                "session_id": session_id,
                "step_name": step_name,
                "correlation_id": f"{session_id}:{step_name}:run-{uuid4()}",
                "state": session["state"],
                "policy_profile": session["policy_profile"],
                "pressure_level": pressure,
                "fallback_step": session["fallback_current_step"],
                "fallback_result": fallback_result,
            }
            if prompt is not None:
                result["prompt"] = prompt
                result["generation"] = {
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "runtime_backend": session.get("runtime_backend"),
                    "runtime_profile": session.get("runtime_profile"),
                    "runtime_backend_options": session.get("runtime_backend_options", {}),
                    "runtime_profile_override": normalized_profile_override,
                    "runtime_backend_options_override": runtime_backend_options_override,
                }
                result["inference_result"] = {
                    "backend": session.get("runtime_backend") or "simulation",
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "runtime_profile": session.get("runtime_profile"),
                    "runtime_backend_options": session.get("runtime_backend_options", {}),
                    "runtime_profile_override": normalized_profile_override,
                    "runtime_backend_options_override": runtime_backend_options_override,
                    "text": f"[simulated:{session.get('runtime_backend') or 'default'}] {prompt}",
                }
            if normalized_profile_override is not None:
                result["runtime_profile_override"] = normalized_profile_override
            if runtime_backend_options_override is not None:
                result["runtime_backend_options_override"] = runtime_backend_options_override
            self._save()
            return result
        finally:
            session["active_run"] = False

    def get_residency(self, session_id: str, caller: CallerIdentity) -> dict[str, Any]:
        session = self._get_owned_session(session_id, caller, "GetResidency")
        snap = ResidencySnapshot(
            session_id=session_id,
            session_state=SessionState(session["state"]),
            primary_tier=self._derive_primary_tier(session),
            tensor_residency={name: ResidencyState(tensor["residency"]) for name, tensor in session["tensors"].items()},
            vram_bytes=session["vram_used_bytes"],
            pinned_ram_bytes=session["pinned_ram_used_bytes"],
            pageable_ram_bytes=session["pageable_ram_used_bytes"],
            cpu_only_bytes=session["cpu_only_used_bytes"],
            active_run_in_progress=session["active_run"],
        )
        return _json_safe(snap)

    def get_pressure(self, session_id: str, caller: CallerIdentity) -> dict[str, Any]:
        session = self._get_owned_session(session_id, caller, "GetPressure")
        snap = PressureSnapshot(
            session_id=session_id,
            vram_budget_bytes=session["vram_budget_bytes"],
            vram_used_bytes=session["vram_used_bytes"],
            pinned_ram_used_bytes=session["pinned_ram_used_bytes"],
            pressure_level=self._compute_pressure(session),
            policy_profile=PolicyProfile(session["policy_profile"]),
            timestamp_ms=_now_ms(),
        )
        return _json_safe(snap)

    def set_policy(self, session_id: str, policy: PolicyProfile, caller: CallerIdentity) -> dict[str, Any]:
        session = self._get_owned_session(session_id, caller, "SetPolicy")
        if session["active_run"]:
            raise ApiError(ApiErrorCode.INVALID_STATE, "cannot change policy during RunStep")
        session["policy_profile"] = _to_policy_profile(policy).value
        self._save()
        return {"session_id": session_id, "policy_profile": session["policy_profile"]}

    def close_session(self, session_id: str, caller: CallerIdentity) -> dict[str, Any]:
        session = self._get_session(session_id)
        if session["closed"] or session["state"] == SessionState.CLOSED.value:
            self._save()
            return {"session_id": session_id, "closed": True, "idempotent": True}
        self._authorize(caller)
        if caller.user_sid != session["owner_sid"]:
            raise ApiError(ApiErrorCode.AUTH_DENIED, "CloseSession caller does not own the session")
        session["closed"] = True
        session["state"] = SessionState.CLOSED.value
        session["active_run"] = False
        key = _caller_key(caller)
        self.state["active_sessions"][key] = max(0, self.state["active_sessions"].get(key, 0) - 1)
        if self.state["active_sessions"][key] <= 0:
            self.state["active_sessions"].pop(key, None)
        self._save()
        return {"session_id": session_id, "closed": True, "idempotent": False}

    def _advance_fallback(self, session: dict[str, Any], caller: CallerIdentity) -> dict[str, Any]:
        state = FallbackState(
            current_step=FallbackStep(session["fallback_current_step"]) if session["fallback_current_step"] else None,
            last_step_change_ms=session["fallback_last_step_change_ms"],
            step_change_history_ms=tuple(session["fallback_step_change_history_ms"]),
            stability_mode=session["fallback_stability_mode"],
        )
        decision = self.fallback.evaluate(state, _now_ms())
        if decision.enter_stability_mode:
            session["fallback_stability_mode"] = True
        if not decision.should_advance:
            session["state"] = SessionState.DEGRADED.value
            return {
                "should_advance": False,
                "next_step": session["fallback_current_step"],
                "enter_stability_mode": session["fallback_stability_mode"],
                "reason_code": decision.reason_code,
            }
        next_step = decision.next_step
        if next_step is None or next_step is FallbackStep.CONTROLLED_FAIL:
            session["fallback_current_step"] = FallbackStep.CONTROLLED_FAIL.value
            session["fallback_last_step_change_ms"] = _now_ms()
            session["fallback_step_change_history_ms"].append(session["fallback_last_step_change_ms"])
            session["state"] = SessionState.FAILED.value
            return {"should_advance": True, "next_step": FallbackStep.CONTROLLED_FAIL.value, "enter_stability_mode": session["fallback_stability_mode"], "reason_code": decision.reason_code}
        self._apply_fallback(session, next_step)
        session["fallback_current_step"] = next_step.value
        session["fallback_last_step_change_ms"] = _now_ms()
        session["fallback_step_change_history_ms"].append(session["fallback_last_step_change_ms"])
        session["state"] = SessionState.DEGRADED.value
        return {"should_advance": True, "next_step": next_step.value, "enter_stability_mode": session["fallback_stability_mode"], "reason_code": decision.reason_code}

    def _apply_fallback(self, session: dict[str, Any], next_step: FallbackStep) -> None:
        tensor = self._select_tensor(session)
        if tensor is None:
            return
        if next_step is FallbackStep.KV_CONTEXT_REDUCTION:
            destination = ResidencyState.PINNED_RAM
        elif next_step is FallbackStep.BATCH_REDUCTION:
            destination = ResidencyState.PAGEABLE_RAM
        elif next_step is FallbackStep.PRECISION_REDUCTION:
            destination = ResidencyState.PAGEABLE_RAM
        elif next_step is FallbackStep.SELECTIVE_CPU_OFFLOAD:
            destination = ResidencyState.CPU_ONLY
        else:
            return
        self._migrate_tensor(session, tensor, destination, reason_code=next_step.value)

    def _migrate_tensor(self, session: dict[str, Any], tensor: dict[str, Any], destination: ResidencyState, *, reason_code: str) -> dict[str, Any]:
        source = ResidencyState(tensor["residency"])
        tensor["residency"] = destination.value
        self._apply_bytes(session, tensor["size_bytes"], source, destination)
        return {"tensor_name": tensor["name"], "source": source.value, "destination": destination.value, "bytes_moved": tensor["size_bytes"], "reason_code": reason_code, "timestamp_ms": _now_ms()}

    def _select_tensor(self, session: dict[str, Any]) -> dict[str, Any] | None:
        order = {ResidencyState.VRAM.value: 0, ResidencyState.PINNED_RAM.value: 1, ResidencyState.PAGEABLE_RAM.value: 2, ResidencyState.CPU_ONLY.value: 3}
        tensors = sorted(session["tensors"].values(), key=lambda tensor: (order.get(tensor["residency"], 99), tensor["name"]))
        return tensors[0] if tensors else None

    @staticmethod
    def _apply_bytes(session: dict[str, Any], size_bytes: int, source: ResidencyState, destination: ResidencyState) -> None:
        def dec(state: ResidencyState) -> None:
            key = {
                ResidencyState.VRAM: "vram_used_bytes",
                ResidencyState.PINNED_RAM: "pinned_ram_used_bytes",
                ResidencyState.PAGEABLE_RAM: "pageable_ram_used_bytes",
                ResidencyState.CPU_ONLY: "cpu_only_used_bytes",
            }[state]
            session[key] = max(0, session[key] - size_bytes)
        def inc(state: ResidencyState) -> None:
            key = {
                ResidencyState.VRAM: "vram_used_bytes",
                ResidencyState.PINNED_RAM: "pinned_ram_used_bytes",
                ResidencyState.PAGEABLE_RAM: "pageable_ram_used_bytes",
                ResidencyState.CPU_ONLY: "cpu_only_used_bytes",
            }[state]
            session[key] += size_bytes
        dec(source); inc(destination)

    @staticmethod
    def _derive_primary_tier(session: dict[str, Any]) -> MemoryTier:
        if any(t["residency"] == ResidencyState.VRAM.value for t in session["tensors"].values()):
            return MemoryTier.HOT
        if any(t["residency"] == ResidencyState.PINNED_RAM.value for t in session["tensors"].values()):
            return MemoryTier.WARM
        return MemoryTier.COLD

    @staticmethod
    def _compute_pressure(session: dict[str, Any]) -> float:
        return 1.0 if session["vram_budget_bytes"] <= 0 else max(0.0, min(1.0, session["vram_used_bytes"] / session["vram_budget_bytes"]))

    def _authorize(self, caller: CallerIdentity) -> None:
        if not _valid_caller(caller):
            raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "caller identity is malformed")
        if caller.user_sid != self.state["service_owner_sid"]:
            raise ApiError(ApiErrorCode.AUTH_DENIED, "caller is not authorized for this local endpoint")

    def _record_attempt(self, caller: CallerIdentity, now_ms: int) -> None:
        key = _caller_key(caller)
        self.state["create_session_attempts"].setdefault(key, []).append(now_ms)

    def _enforce_rate_limit(self, caller: CallerIdentity, now_ms: int) -> None:
        key = _caller_key(caller)
        attempts = self.state["create_session_attempts"].setdefault(key, [])
        cutoff = now_ms - 60_000
        # H9 fix: avoid O(n^2) list.pop(0); slice off stale entries in one pass
        first_valid = 0
        while first_valid < len(attempts) and attempts[first_valid] < cutoff:
            first_valid += 1
        if first_valid:
            del attempts[:first_valid]
        if len(attempts) > 6:
            raise ApiError(ApiErrorCode.RATE_LIMITED, "CreateSession rate limit exceeded")

    def _enforce_session_cap(self, caller: CallerIdentity) -> None:
        if self.state["active_sessions"].get(_caller_key(caller), 0) >= 8:
            raise ApiError(ApiErrorCode.RESOURCE_EXHAUSTED, "maximum concurrent sessions reached for caller")

    def _get_session(self, session_id: str) -> dict[str, Any]:
        session = self.state["sessions"].get(session_id)
        if session is None:
            raise ApiError(ApiErrorCode.NOT_FOUND, "session not found")
        return session

    def _get_owned_session(self, session_id: str, caller: CallerIdentity, api_name: str) -> dict[str, Any]:
        session = self._get_session(session_id)
        if session["closed"] or session["state"] == SessionState.CLOSED.value:
            raise ApiError(ApiErrorCode.INVALID_STATE, "session is closed")
        self._authorize(caller)
        if caller.user_sid != session["owner_sid"]:
            raise ApiError(ApiErrorCode.AUTH_DENIED, f"{api_name} caller does not own the session")
        return session

    @staticmethod
    def _require_state(session: dict[str, Any], allowed: set[SessionState], api_name: str) -> None:
        if SessionState(session["state"]) not in allowed:
            allowed_states = ", ".join(state.value for state in allowed)
            raise ApiError(ApiErrorCode.INVALID_STATE, f"{api_name} requires one of: {allowed_states}; current state={session['state']}")

    def _get_tensor(self, session: dict[str, Any], tensor_name: str) -> dict[str, Any]:
        tensors = session["tensors"]
        if tensor_name not in tensors:
            raise ApiError(ApiErrorCode.NOT_FOUND, "tensor not registered")
        return tensors[tensor_name]

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return _default_state()
        with self.path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if int(payload.get("version", 0)) != STATE_VERSION:
            raise ApiError(ApiErrorCode.INTERNAL, "unsupported CLI state version")
        # C3: verify HMAC integrity tag before trusting the content
        stored_hmac = payload.pop("_hmac", None)
        if stored_hmac is not None:
            json_bytes = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
            expected_hmac = _compute_state_hmac(json_bytes)
            if not hmac.compare_digest(stored_hmac, expected_hmac):
                raise ApiError(ApiErrorCode.INTERNAL, "CLI state file integrity check failed")
        return payload

    @staticmethod
    def _lock_file(f: Any) -> None:
        """Acquire an exclusive file lock (best-effort on non-Windows)."""
        try:
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except (ImportError, OSError):
            # On non-Windows or if locking fails, proceed without lock.
            pass

    @staticmethod
    def _unlock_file(f: Any) -> None:
        """Release the file lock (best-effort)."""
        try:
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        except (ImportError, OSError):
            pass

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        # M4: use file-based locking to prevent race conditions
        lock_path = self.path.with_suffix(".lock")
        lock_fh = None
        try:
            lock_fh = lock_path.open("w", encoding="utf-8")
            self._lock_file(lock_fh)
            # C3: compute HMAC integrity tag and embed in state JSON
            payload = dict(self.state)
            json_bytes = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
            payload["_hmac"] = _compute_state_hmac(json_bytes)
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
            tmp.replace(self.path)
        finally:
            if lock_fh is not None:
                self._unlock_file(lock_fh)
                lock_fh.close()


class RemoteBackend:
    """Thin adapter over the SDK when an IPC endpoint is available."""

    def __init__(self, endpoint: str, caller: CallerIdentity) -> None:
        if AstraWeaveSDK is None:
            raise ApiError(ApiErrorCode.INTERNAL, "AstraWeaveSDK transport is unavailable")
        self.sdk = AstraWeaveSDK(endpoint=endpoint, default_caller_identity=caller)

    def create_session(self, caller: CallerIdentity) -> dict[str, Any]:
        return {"session_id": self.sdk.CreateSession(caller_identity=caller)}

    def load_model(
        self,
        session_id: str,
        model_name: str,
        caller: CallerIdentity,
        runtime_backend: str | None = None,
        runtime_profile: str | None = None,
        runtime_backend_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.sdk.LoadModel(
            session_id,
            model_name,
            runtime_backend=runtime_backend,
            runtime_profile=runtime_profile,
            runtime_backend_options=runtime_backend_options,
            caller_identity=caller,
        )
        result = {"session_id": session_id, "model_name": model_name}
        if runtime_backend is not None:
            result["runtime_backend"] = runtime_backend
        if runtime_profile is not None:
            result["runtime_profile"] = runtime_profile
        if runtime_backend_options is not None:
            result["runtime_backend_options"] = runtime_backend_options
        return result

    def register_tensor(self, session_id: str, tensor_name: str, size_bytes: int, caller: CallerIdentity) -> dict[str, Any]:
        self.sdk.RegisterTensor(session_id, tensor_name, size_bytes, caller_identity=caller)
        return {"session_id": session_id, "tensor_name": tensor_name, "size_bytes": size_bytes}

    def set_tier_hint(self, session_id: str, tensor_name: str, tier: MemoryTier, caller: CallerIdentity) -> dict[str, Any]:
        self.sdk.SetTierHint(session_id, tensor_name, tier, caller_identity=caller)
        return {"session_id": session_id, "tensor_name": tensor_name, "tier_hint": tier.value}

    def prefetch_plan(self, session_id: str, caller: CallerIdentity) -> Any:
        return self.sdk.PrefetchPlan(session_id, caller_identity=caller)

    def run_step(
        self,
        session_id: str,
        step_name: str,
        caller: CallerIdentity,
        prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        runtime_profile_override: str | None = None,
        runtime_backend_options_override: dict[str, Any] | None = None,
    ) -> Any:
        return self.sdk.RunStep(
            session_id,
            step_name=step_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            runtime_profile_override=runtime_profile_override,
            runtime_backend_options_override=runtime_backend_options_override,
            caller_identity=caller,
        )

    def get_residency(self, session_id: str, caller: CallerIdentity) -> Any:
        return self.sdk.GetResidency(session_id, caller_identity=caller)

    def get_pressure(self, session_id: str, caller: CallerIdentity) -> Any:
        return self.sdk.GetPressure(session_id, caller_identity=caller)

    def set_policy(self, session_id: str, policy: PolicyProfile, caller: CallerIdentity) -> dict[str, Any]:
        self.sdk.SetPolicy(session_id, policy, caller_identity=caller)
        return {"session_id": session_id, "policy_profile": policy.value}

    def close_session(self, session_id: str, caller: CallerIdentity) -> dict[str, Any]:
        self.sdk.CloseSession(session_id, caller_identity=caller)
        return {"session_id": session_id, "closed": True, "idempotent": False}


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    common.add_argument("--backend", choices=["auto", "remote", "local"], default="auto")
    common.add_argument("--caller-sid", default=DEFAULT_SERVICE_OWNER_SID)
    common.add_argument("--caller-pid", type=int, default=None)

    parser = argparse.ArgumentParser(prog="astrawave", description="AstraWeave CLI", parents=[common])
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("create-session")

    p = sub.add_parser("load-model")
    p.add_argument("session_id")
    p.add_argument("model_name")
    p.add_argument(
        "--runtime-backend",
        default=None,
        help="Optional runtime backend selector for the loaded model (for example: auto, simulation, ollama).",
    )
    p.add_argument(
        "--runtime-profile",
        default=None,
        help="Optional load-time tuning profile (for example: auto or vram_constrained).",
    )
    p.add_argument(
        "--runtime-option",
        dest="runtime_options",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Optional load-time backend tuning option; may be repeated.",
    )

    p = sub.add_parser("register-tensor")
    p.add_argument("session_id")
    p.add_argument("tensor_name")
    p.add_argument("size_bytes", type=int)

    p = sub.add_parser("set-tier-hint")
    p.add_argument("session_id")
    p.add_argument("tensor_name")
    p.add_argument("tier", choices=[item.value for item in MemoryTier])

    p = sub.add_parser("prefetch-plan")
    p.add_argument("session_id")

    p = sub.add_parser("run-step")
    p.add_argument("session_id")
    p.add_argument("--step-name", default="run")
    p.add_argument("--prompt", default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument(
        "--runtime-profile-override",
        default=None,
        help="Optional per-step tuning profile override.",
    )
    p.add_argument(
        "--runtime-option-override",
        dest="runtime_option_overrides",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Optional per-step backend tuning option override; may be repeated.",
    )

    p = sub.add_parser("get-residency")
    p.add_argument("session_id")

    p = sub.add_parser("get-pressure")
    p.add_argument("session_id")

    p = sub.add_parser("set-policy")
    p.add_argument("session_id")
    p.add_argument("policy", choices=[item.value for item in PolicyProfile])

    p = sub.add_parser("close-session")
    p.add_argument("session_id")

    sub.add_parser("hardware-probe")

    serve = sub.add_parser("serve")
    serve.add_argument(
        "--endpoint",
        "--serve-endpoint",
        dest="serve_endpoint",
        default=DEFAULT_SERVE_ENDPOINT,
    )
    serve.add_argument("--transport", choices=["auto", "tcp", "pipe"], default="auto")
    serve.add_argument("--duration-seconds", type=float, default=None)
    return parser


def _dispatch(backend: _Backend, args: argparse.Namespace, caller: CallerIdentity) -> Any:
    if args.command == "create-session":
        return backend.create_session(caller)
    if args.command == "load-model":
        runtime_backend_options = _parse_runtime_option_pairs(args.runtime_options, field_name="runtime_options")
        return backend.load_model(
            args.session_id,
            args.model_name,
            caller,
            args.runtime_backend,
            args.runtime_profile,
            runtime_backend_options,
        )
    if args.command == "register-tensor":
        return backend.register_tensor(args.session_id, args.tensor_name, args.size_bytes, caller)
    if args.command == "set-tier-hint":
        return backend.set_tier_hint(args.session_id, args.tensor_name, _to_memory_tier(args.tier), caller)
    if args.command == "prefetch-plan":
        return backend.prefetch_plan(args.session_id, caller)
    if args.command == "run-step":
        runtime_backend_options_override = _parse_runtime_option_pairs(
            args.runtime_option_overrides,
            field_name="runtime_option_overrides",
        )
        return backend.run_step(
            args.session_id,
            args.step_name,
            caller,
            args.prompt,
            args.max_tokens,
            args.temperature,
            args.runtime_profile_override,
            runtime_backend_options_override,
        )
    if args.command == "get-residency":
        return backend.get_residency(args.session_id, caller)
    if args.command == "get-pressure":
        return backend.get_pressure(args.session_id, caller)
    if args.command == "set-policy":
        return backend.set_policy(args.session_id, _to_policy_profile(args.policy), caller)
    if args.command == "close-session":
        return backend.close_session(args.session_id, caller)
    raise ApiError(ApiErrorCode.INVALID_ARGUMENT, f"unknown command {args.command}")


def _serve(args: argparse.Namespace) -> int:
    duration_seconds = args.duration_seconds
    if AstraWeaveServiceHost is not None and ServiceHostConfig is not None:
        transport_config = _parse_serve_endpoint(args.serve_endpoint, args.transport)
        config = ServiceHostConfig(
            endpoint=None,
            prefer_named_pipe=bool(transport_config.get("prefer_named_pipe", False)),
            pipe_name=str(transport_config.get("pipe_name", r"\\.\pipe\astrawave")),
            host=str(transport_config.get("host", "127.0.0.1")),
            port=int(transport_config.get("port", 0)),
        )
        host = AstraWeaveServiceHost(config)
        started = False
        try:
            status = host.start()
            started = True
            _emit_ok(
                {
                    "stage": "started",
                    "requested_endpoint": args.serve_endpoint,
                    "requested_transport": args.transport,
                    "endpoint": _describe_endpoint(status.endpoint),
                    "transport": status.transport,
                    "pid": os.getpid(),
                    "duration_seconds": duration_seconds,
                }
            )

            if duration_seconds is None:
                while host.is_running:
                    sleep(0.2)
            else:
                host.run_for(max(duration_seconds, 0.0))
        except KeyboardInterrupt:
            return 130
        except ApiError:
            raise
        except Exception as exc:
            raise ApiError(ApiErrorCode.INTERNAL, "failed to start AstraWeave service host") from exc
        finally:
            if started:
                stopped = host.stop()
                _emit_ok(
                    {
                        "stage": "stopped",
                        "endpoint": _describe_endpoint(stopped.endpoint),
                        "transport": stopped.transport,
                        "duration_seconds": duration_seconds,
                    }
                )
        return 0

    if AstraWeaveIpcServer is None:
        raise ApiError(ApiErrorCode.INTERNAL, "AstraWeaveIpcServer transport is unavailable")

    config = _parse_serve_endpoint(args.serve_endpoint, args.transport)
    server = AstraWeaveIpcServer(**config)
    started = False
    exit_code = 0
    try:
        server.start()
        started = True
        _emit_ok(
            {
                "stage": "started",
                "requested_endpoint": args.serve_endpoint,
                "requested_transport": args.transport,
                "endpoint": _describe_endpoint(server.endpoint),
                "transport": server.transport,
                "pid": os.getpid(),
                "duration_seconds": duration_seconds,
            }
        )

        if duration_seconds is None:
            while True:
                sleep(0.2)
        else:
            deadline = monotonic() + max(duration_seconds, 0.0)
            while True:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    break
                sleep(min(0.2, remaining))
    except KeyboardInterrupt:
        exit_code = 130
    except ApiError:
        raise
    except Exception as exc:
        raise ApiError(ApiErrorCode.INTERNAL, "failed to start AstraWeave service host") from exc
    finally:
        if started:
            with suppress(Exception):
                server.stop()
            _emit_ok(
                {
                    "stage": "stopped",
                    "endpoint": _describe_endpoint(server.endpoint),
                    "transport": server.transport,
                    "duration_seconds": duration_seconds,
                }
            )
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "serve":
        try:
            return _serve(args)
        except ApiError as exc:
            _emit_error(exc)
            return 1
    if args.command == "hardware-probe":
        try:
            _emit_ok(_json_safe(_hardware_probe_result()))
            return 0
        except ApiError as exc:
            _emit_error(exc)
            return 1
    caller = CallerIdentity(user_sid=args.caller_sid, pid=args.caller_pid if args.caller_pid is not None else os.getpid())
    try:
        backend = _resolve_backend(args.endpoint, caller, args.backend)
        result = _dispatch(backend, args, caller)
        _emit_ok(_json_safe(result))
        return 0
    except ApiError as exc:
        _emit_error(exc)
        return 1


def _resolve_backend(endpoint: str, caller: CallerIdentity, mode: str = "auto") -> Any:
    endpoint = (endpoint or "").strip()
    mode = (mode or "auto").strip().lower()
    if mode not in {"auto", "remote", "local"}:
        raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "backend must be one of: auto, remote, local")

    if mode == "local":
        return LocalBackend(endpoint or "local://default")
    if mode == "remote":
        return RemoteBackend(endpoint or "auto", caller)

    remote_hint = endpoint.startswith(("tcp://", "pipe://", "\\\\.\\pipe\\"))
    local_hint = endpoint.startswith("local://")
    if local_hint:
        return LocalBackend(endpoint)

    if endpoint == "auto" or remote_hint or endpoint == "":
        try:
            return RemoteBackend(endpoint or "auto", caller)
        except (ConnectionError, OSError):
            # M6: only catch connection-related errors so the fallback to
            # LocalBackend is reachable; let ApiError propagate directly.
            if remote_hint:
                raise
    return LocalBackend(endpoint or DEFAULT_ENDPOINT)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
