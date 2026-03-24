"""In-memory AstraWeave service prototype with integrated runtime modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from inspect import Parameter, signature
import os
from threading import RLock
from time import time_ns
from typing import Any, Callable, Dict, Iterable, Mapping, Optional
from uuid import uuid4

try:
    from .cuda_runtime import DEFAULT_DEVICE_INDEX, DEFAULT_TRANSFER_BYTES, run_cuda_transfer as _run_cuda_transfer
except Exception:  # pragma: no cover - optional runtime path
    DEFAULT_DEVICE_INDEX = 0
    DEFAULT_TRANSFER_BYTES = 1_048_576
    _run_cuda_transfer = None

from .errors import ApiError, ApiErrorCode
from .fallback import FallbackController, FallbackState, FallbackStep
from .inference_runtime import (
    InferenceRuntime,
    create_inference_runtime,
    resolve_backend_and_model_name,
)
from .runtime_tuning import merge_backend_options, resolve_runtime_tuning
from .security import CallerIdentity, SecurityGuard, SecurityPolicy, resolve_current_user_sid
from .telemetry import (
    FallbackEvent as TelemetryFallbackEvent,
    TelemetryEvent,
    TelemetryEventType,
    TelemetryPipeline,
    TelemetryPolicy,
    TelemetryReasonCode,
    TransferEvent as TelemetryTransferEvent,
    new_correlation_id,
)
from .types import (
    FallbackEvent,
    MemoryTier,
    PolicyProfile,
    PressureSnapshot,
    ResidencySnapshot,
    ResidencyState,
    SessionState,
    TransferEvent,
)

HIGH_PRESSURE_THRESHOLD = 0.75
DEFAULT_SERVICE_OWNER_SID = resolve_current_user_sid() or "S-1-5-21-AstraWeave-Owner"
DEFAULT_RUNSTEP_MODE = "auto"
RUNSTEP_MODE_ENV = "ASTRAWEAVE_RUNSTEP_MODE"
RUNSTEP_MODE_ENABLE_ENV = "ASTRAWEAVE_ENABLE_HARDWARE_RUNSTEP"
HARDWARE_TRANSFER_BYTES_ENV = "ASTRAWEAVE_HARDWARE_TRANSFER_BYTES"
HARDWARE_DEVICE_INDEX_ENV = "ASTRAWEAVE_HARDWARE_DEVICE_INDEX"
HARDWARE_HOLD_MS_ENV = "ASTRAWEAVE_HARDWARE_HOLD_MS"
DEFAULT_HARDWARE_HOLD_MS = 75

HardwareExecutor = Callable[..., dict[str, Any]]
InferenceRuntimeFactory = Callable[[str], InferenceRuntime]


@dataclass(frozen=True, slots=True, kw_only=True)
class _SessionLifecycleEvent(TelemetryEvent):
    """Internal telemetry event for session lifecycle boundaries."""

    event_type = TelemetryEventType.CUSTOM
    action: str
    detail: str | None = None


@dataclass(slots=True)
class _TensorRecord:
    """Tracked tensor metadata used by the service prototype."""

    name: str
    size_bytes: int
    tier_hint: MemoryTier = MemoryTier.WARM
    residency: ResidencyState = ResidencyState.PINNED_RAM


@dataclass(slots=True)
class _Session:
    """Tracked in-memory session state."""

    session_id: str
    owner_identity: CallerIdentity
    state: SessionState = SessionState.SESSION_CREATED
    model_name: Optional[str] = None
    inference_backend: str = "simulation"
    resolved_model_name: Optional[str] = None
    inference_metadata: dict[str, Any] = field(default_factory=dict)
    policy_profile: PolicyProfile = PolicyProfile.STABILITY
    tensors: Dict[str, _TensorRecord] = field(default_factory=dict)
    active_run: bool = False
    closed: bool = False
    fallback_current_step: FallbackStep | None = None
    fallback_last_step_change_ms: int | None = None
    fallback_step_change_history_ms: tuple[int, ...] = ()
    fallback_stability_mode: bool = False
    fallback_events: list[FallbackEvent] = field(default_factory=list)
    transfer_history: list[TransferEvent] = field(default_factory=list)
    lock: RLock = field(default_factory=RLock)
    vram_budget_bytes: int = 8 * 1024**3
    vram_used_bytes: int = 0
    pinned_ram_used_bytes: int = 0
    pageable_ram_used_bytes: int = 0
    cpu_only_used_bytes: int = 0


class AstraWeaveService:
    """Prototype implementation of AstraWeave service APIs.

    The service enforces the lifecycle described in `docs/api-contract.md`:

    `CreateSession -> LoadModel -> RegisterTensor/SetTierHint/PrefetchPlan -> RunStep -> CloseSession`

    It integrates the existing support modules:
    - `SecurityGuard` for caller admission and ownership checks
    - `TelemetryPipeline` for local-only structured telemetry
    - `FallbackController` for deterministic pressure-driven fallback
    """

    def __init__(
        self,
        *,
        security_guard: SecurityGuard | None = None,
        telemetry_pipeline: TelemetryPipeline | None = None,
        fallback_controller: FallbackController | None = None,
        runstep_mode: str | None = None,
        hardware_executor: HardwareExecutor | None = None,
        inference_runtime_factory: InferenceRuntimeFactory | None = None,
    ) -> None:
        self._security = security_guard or SecurityGuard(
            SecurityPolicy(service_owner_sid=DEFAULT_SERVICE_OWNER_SID)
        )
        self._telemetry = telemetry_pipeline or TelemetryPipeline(
            policy=TelemetryPolicy(local_only=True, export_opt_in=False)
        )
        self._fallback_controller = fallback_controller or FallbackController()
        self._service_owner_sid = self._security.policy.service_owner_sid
        self._sessions: Dict[str, _Session] = {}
        self._closed_sessions: Dict[str, _Session] = {}
        self._lock = RLock()
        self._runstep_mode_override = runstep_mode
        self._hardware_executor = hardware_executor or _run_cuda_transfer
        self._inference_runtime_factory = inference_runtime_factory or create_inference_runtime

    @property
    def telemetry(self) -> TelemetryPipeline:
        """Expose the telemetry pipeline for tests and diagnostics."""

        return self._telemetry

    @property
    def security_guard(self) -> SecurityGuard:
        """Expose the security guard for tests and diagnostics."""

        return self._security

    @property
    def fallback_controller(self) -> FallbackController:
        """Expose the fallback controller for tests and diagnostics."""

        return self._fallback_controller

    def CreateSession(self, caller_identity: CallerIdentity | None = None) -> str:
        """Create a new session and return its opaque identifier."""

        caller = self._resolve_creation_caller(caller_identity)
        admission = self._security.admit_create_session(caller)
        if not admission.allowed:
            raise admission.to_api_error()

        session_id = str(uuid4())
        session = _Session(session_id=session_id, owner_identity=caller)
        with self._lock:
            self._sessions[session_id] = session

        try:
            self._record_lifecycle_event(
                session=session,
                caller=caller,
                action="create_session",
                detail="session_created",
            )
        except Exception:
            with self._lock:
                self._sessions.pop(session_id, None)
            self._security.release_session(caller)
            raise

        return session_id

    def LoadModel(
        self,
        session_id: str,
        model_name: str,
        *,
        runtime_backend: str | None = None,
        runtime_profile: str | None = None,
        runtime_backend_options: Mapping[str, Any] | None = None,
        backend_options: Mapping[str, Any] | None = None,
        caller_identity: CallerIdentity | None = None,
    ) -> None:
        """Attach a model to a session and transition it toward READY."""

        session, _caller = self._get_authorized_session(session_id, caller_identity, "LoadModel")
        self._require_state(session, {SessionState.SESSION_CREATED}, "LoadModel")
        backend_name, resolved_model_name = resolve_backend_and_model_name(
            model_name,
            runtime_backend=runtime_backend,
        )
        tuning = resolve_runtime_tuning(
            resolved_model_name,
            runtime_profile=runtime_profile,
            backend_options=merge_backend_options(backend_options, runtime_backend_options),
        )
        runtime = self._inference_runtime_factory(backend_name)
        binding = self._call_runtime_method(
            runtime,
            "load_model",
            resolved_model_name,
            runtime_profile=tuning.profile_name,
            backend_options=tuning.backend_options,
        )
        with session.lock:
            session.model_name = model_name
            session.inference_backend = binding.backend
            session.resolved_model_name = binding.resolved_model_name
            session.inference_metadata = self._build_inference_metadata(
                binding.metadata,
                requested_model_name=binding.requested_model_name,
                runtime_profile=tuning.profile_name,
                model_size_billion=tuning.model_size_billion,
                model_size_label=tuning.model_size_label,
                backend_options=tuning.backend_options,
            )
            session.state = SessionState.MODEL_LOADED
            session.state = SessionState.READY

    def RegisterTensor(
        self,
        session_id: str,
        tensor_name: str,
        size_bytes: int,
        *,
        caller_identity: CallerIdentity | None = None,
    ) -> None:
        """Register a tensor and initialize its residency metadata."""

        session, _caller = self._get_authorized_session(session_id, caller_identity, "RegisterTensor")
        self._require_state(
            session,
            {SessionState.MODEL_LOADED, SessionState.READY, SessionState.DEGRADED},
            "RegisterTensor",
        )
        if not tensor_name:
            raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "tensor_name must not be empty")
        if size_bytes <= 0:
            raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "size_bytes must be positive")
        with session.lock:
            if tensor_name in session.tensors:
                raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "tensor already registered")
            session.tensors[tensor_name] = _TensorRecord(
                name=tensor_name,
                size_bytes=size_bytes,
                tier_hint=MemoryTier.WARM,
                residency=ResidencyState.PINNED_RAM,
            )
            session.pinned_ram_used_bytes += size_bytes
            session.state = SessionState.READY

    def SetTierHint(
        self,
        session_id: str,
        tensor_name: str,
        tier: MemoryTier,
        *,
        caller_identity: CallerIdentity | None = None,
    ) -> None:
        """Update the tier hint for a previously registered tensor."""

        session, _caller = self._get_authorized_session(session_id, caller_identity, "SetTierHint")
        self._require_state(
            session,
            {SessionState.MODEL_LOADED, SessionState.READY, SessionState.DEGRADED},
            "SetTierHint",
        )
        with session.lock:
            tensor = self._get_tensor(session, tensor_name)
            tensor.tier_hint = self._normalize_tier(tier)

    def PrefetchPlan(
        self,
        session_id: str,
        *,
        caller_identity: CallerIdentity | None = None,
    ) -> list[TransferEvent]:
        """Generate a simple prefetch plan for the session."""

        session, caller = self._get_authorized_session(session_id, caller_identity, "PrefetchPlan")
        self._require_state(session, {SessionState.READY, SessionState.DEGRADED}, "PrefetchPlan")
        with session.lock:
            plan: list[TransferEvent] = []
            for tensor in sorted(session.tensors.values(), key=lambda item: item.name):
                destination = self._residency_for_tier(tensor.tier_hint)
                if tensor.residency != destination:
                    event = self._migrate_tensor(
                        session=session,
                        tensor=tensor,
                        destination=destination,
                        reason_code="prefetch",
                        telemetry_reason=TelemetryReasonCode.TRANSFER_PREFETCH,
                        telemetry_mode=session.policy_profile.value,
                        telemetry_direction="prefetch",
                        caller=caller,
                    )
                    plan.append(event)
            return plan

    def RunStep(
        self,
        session_id: str,
        step_name: str = "run",
        *,
        prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system_prompt: str | None = None,
        runtime_profile_override: str | None = None,
        runtime_backend_options_override: Mapping[str, Any] | None = None,
        backend_options: Mapping[str, Any] | None = None,
        caller_identity: CallerIdentity | None = None,
    ) -> dict[str, object]:
        """Execute one inference step with single-flight protection per session."""

        session, caller = self._get_authorized_session(session_id, caller_identity, "RunStep")
        self._require_state(session, {SessionState.READY, SessionState.DEGRADED}, "RunStep")
        if not step_name:
            raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "step_name must not be empty")

        with session.lock:
            if session.active_run:
                raise ApiError(ApiErrorCode.CONFLICT_RUN_IN_PROGRESS, "a RunStep is already active")
            session.active_run = True
            session.state = SessionState.RUNNING

        try:
            now_ms = self._now_ms()
            pressure_level = self._compute_pressure_level(session)
            fallback_result: dict[str, object] | None = None
            requested_run_mode = self._resolve_runstep_mode()
            run_mode = "simulation"
            hardware_result: dict[str, object] | None = None
            inference_result: dict[str, object] | None = None

            if pressure_level >= HIGH_PRESSURE_THRESHOLD or session.fallback_stability_mode:
                fallback_result = self._advance_fallback(session=session, caller=caller, now_ms=now_ms)
            if requested_run_mode in {"auto", "hardware"}:
                hardware_result = self._execute_hardware_runstep(session=session, step_name=step_name)
                if isinstance(hardware_result, dict) and bool(hardware_result.get("ok")):
                    run_mode = "hardware"
                elif isinstance(hardware_result, dict):
                    hardware_result.setdefault("fallback_to_simulation", True)
            if prompt is not None:
                effective_backend_options = self._resolve_effective_backend_options(
                    session=session,
                    runtime_profile_override=runtime_profile_override,
                    runtime_backend_options_override=runtime_backend_options_override,
                    backend_options=backend_options,
                )
                inference_result = self._execute_inference_runstep(
                    session=session,
                    step_name=step_name,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system_prompt=system_prompt,
                    backend_options=effective_backend_options,
                )

            with session.lock:
                if session.state != SessionState.FAILED:
                    if (
                        requested_run_mode in {"auto", "hardware"}
                        and isinstance(hardware_result, dict)
                        and not bool(hardware_result.get("ok"))
                    ):
                        session.state = SessionState.DEGRADED
                    elif isinstance(inference_result, dict) and not bool(inference_result.get("ok", True)):
                        session.state = SessionState.DEGRADED
                    else:
                        session.state = SessionState.DEGRADED if (
                            session.fallback_stability_mode
                            or session.fallback_current_step is not None
                            or pressure_level >= HIGH_PRESSURE_THRESHOLD
                        ) else SessionState.READY

                correlation_id = f"{session.session_id}:{step_name}:{new_correlation_id('run')}"
                response: dict[str, object] = {
                    "session_id": session.session_id,
                    "step_name": step_name,
                    "correlation_id": correlation_id,
                    "state": session.state,
                    "policy_profile": session.policy_profile,
                    "pressure_level": pressure_level,
                    "fallback_step": session.fallback_current_step.value
                    if session.fallback_current_step is not None
                    else None,
                    "fallback_result": fallback_result,
                    "requested_run_mode": requested_run_mode,
                    "run_mode": run_mode,
                    "hardware_result": hardware_result,
                }
                if prompt is not None:
                    response["inference_result"] = inference_result
                return response
        finally:
            with session.lock:
                session.active_run = False

    def GetResidency(
        self,
        session_id: str,
        *,
        caller_identity: CallerIdentity | None = None,
    ) -> ResidencySnapshot:
        """Return a session-scoped residency snapshot."""

        session, _caller = self._get_authorized_session(session_id, caller_identity, "GetResidency")
        with session.lock:
            tensor_residency = {name: tensor.residency for name, tensor in session.tensors.items()}
            primary_tier = self._derive_primary_tier(session)
            return ResidencySnapshot(
                session_id=session.session_id,
                session_state=session.state,
                primary_tier=primary_tier,
                tensor_residency=tensor_residency,
                vram_bytes=session.vram_used_bytes,
                pinned_ram_bytes=session.pinned_ram_used_bytes,
                pageable_ram_bytes=session.pageable_ram_used_bytes,
                cpu_only_bytes=session.cpu_only_used_bytes,
                active_run_in_progress=session.active_run,
            )

    def GetPressure(
        self,
        session_id: str,
        *,
        caller_identity: CallerIdentity | None = None,
    ) -> PressureSnapshot:
        """Return the current pressure snapshot for a session."""

        session, _caller = self._get_authorized_session(session_id, caller_identity, "GetPressure")
        with session.lock:
            pressure_level = self._compute_pressure_level(session)
            return PressureSnapshot(
                session_id=session.session_id,
                vram_budget_bytes=session.vram_budget_bytes,
                vram_used_bytes=session.vram_used_bytes,
                pinned_ram_used_bytes=session.pinned_ram_used_bytes,
                pressure_level=pressure_level,
                policy_profile=session.policy_profile,
                timestamp_ms=self._now_ms(),
            )

    def SetPolicy(
        self,
        session_id: str,
        policy: PolicyProfile,
        *,
        caller_identity: CallerIdentity | None = None,
    ) -> None:
        """Switch the session policy profile."""

        session, _caller = self._get_authorized_session(session_id, caller_identity, "SetPolicy")
        with session.lock:
            if session.active_run:
                raise ApiError(ApiErrorCode.INVALID_STATE, "cannot change policy during RunStep")
            session.policy_profile = self._normalize_policy(policy)

    def CloseSession(
        self,
        session_id: str,
        *,
        caller_identity: CallerIdentity | None = None,
    ) -> None:
        """Close a session. Closing a known session is idempotent."""

        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = self._closed_sessions.get(session_id)
                if session is None:
                    raise ApiError(ApiErrorCode.NOT_FOUND, "session not found")
                _caller = self._resolve_session_caller(session, caller_identity, "CloseSession")
                return

        caller = self._resolve_session_caller(session, caller_identity, "CloseSession")
        was_open = False

        with session.lock:
            if not session.closed:
                was_open = True
                session.state = SessionState.CLOSED
                session.closed = True
                session.active_run = False

        if was_open:
            self._record_lifecycle_event(
                session=session,
                caller=caller,
                action="close_session",
                detail="session_closed",
            )

            with self._lock:
                self._closed_sessions[session_id] = session
                self._sessions.pop(session_id, None)
            self._security.release_session(session.owner_identity)

    def _resolve_creation_caller(self, caller_identity: CallerIdentity | None) -> CallerIdentity:
        if caller_identity is None:
            return CallerIdentity(self._service_owner_sid, os.getpid())
        if not self._is_valid_caller_identity(caller_identity):
            raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "caller identity is malformed")
        return caller_identity

    def _resolve_session_caller(
        self,
        session: _Session,
        caller_identity: CallerIdentity | None,
        api_name: str,
    ) -> CallerIdentity:
        caller = session.owner_identity if caller_identity is None else caller_identity
        if not self._is_valid_caller_identity(caller):
            raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "caller identity is malformed")

        admission = self._security.authorize_caller(caller)
        if not admission.allowed:
            raise admission.to_api_error()
        if caller.user_sid != session.owner_identity.user_sid:
            raise ApiError(ApiErrorCode.AUTH_DENIED, f"{api_name} caller does not own the session")
        return caller

    def _get_authorized_session(
        self,
        session_id: str,
        caller_identity: CallerIdentity | None,
        api_name: str,
    ) -> tuple[_Session, CallerIdentity]:
        session = self._get_session(session_id)
        if session.closed or session.state == SessionState.CLOSED:
            raise ApiError(ApiErrorCode.INVALID_STATE, "session is closed")
        caller = self._resolve_session_caller(session, caller_identity, api_name)
        return session, caller

    def _get_session(self, session_id: str) -> _Session:
        with self._lock:
            session = self._sessions.get(session_id) or self._closed_sessions.get(session_id)
            if session is None:
                raise ApiError(ApiErrorCode.NOT_FOUND, "session not found")
            return session

    @staticmethod
    def _require_state(session: _Session, allowed: Iterable[SessionState], api_name: str) -> None:
        if session.state not in allowed:
            allowed_states = ", ".join(state.value for state in allowed)
            raise ApiError(
                ApiErrorCode.INVALID_STATE,
                f"{api_name} requires one of: {allowed_states}; current state={session.state.value}",
            )

    @staticmethod
    def _normalize_tier(tier: MemoryTier) -> MemoryTier:
        if not isinstance(tier, MemoryTier):
            raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "tier must be a MemoryTier")
        return tier

    @staticmethod
    def _normalize_policy(policy: PolicyProfile) -> PolicyProfile:
        if not isinstance(policy, PolicyProfile):
            raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "policy must be a PolicyProfile")
        return policy

    @staticmethod
    def _residency_for_tier(tier: MemoryTier) -> ResidencyState:
        if tier is MemoryTier.HOT:
            return ResidencyState.VRAM
        if tier is MemoryTier.WARM:
            return ResidencyState.PINNED_RAM
        return ResidencyState.PAGEABLE_RAM

    def _get_tensor(self, session: _Session, tensor_name: str) -> _TensorRecord:
        if tensor_name not in session.tensors:
            raise ApiError(ApiErrorCode.NOT_FOUND, "tensor not registered")
        return session.tensors[tensor_name]

    def _migrate_tensor(
        self,
        session: _Session,
        tensor: _TensorRecord,
        destination: ResidencyState,
        *,
        reason_code: str,
        telemetry_reason: TelemetryReasonCode,
        telemetry_direction: str,
        telemetry_mode: str,
        caller: CallerIdentity | None = None,
    ) -> TransferEvent:
        source = tensor.residency
        tensor.residency = destination
        self._apply_residency_bytes(session, tensor.size_bytes, source, destination)
        event = TransferEvent(
            session_id=session.session_id,
            source=source,
            destination=destination,
            bytes_moved=tensor.size_bytes,
            reason_code=reason_code,
            timestamp_ms=self._now_ms(),
        )
        session.transfer_history.append(event)
        self._telemetry.record_event(
            TelemetryTransferEvent(
                reason_code=telemetry_reason,
                session_id=session.session_id,
                bytes_moved=tensor.size_bytes,
                direction=telemetry_direction,
                latency_ms=self._estimate_transfer_latency_ms(tensor.size_bytes),
                mode=telemetry_mode,
            ),
            extra_identifiers={
                "tensor_name": tensor.name,
                "source": source.value,
                "destination": destination.value,
                **(
                    {
                        "caller_sid": caller.user_sid,
                        "caller_pid": caller.pid,
                    }
                    if caller is not None
                    else {}
                ),
            },
        )
        return event

    def _record_lifecycle_event(
        self,
        session: _Session,
        caller: CallerIdentity,
        action: str,
        detail: str,
    ) -> None:
        self._telemetry.record_event(
            _SessionLifecycleEvent(
                reason_code=TelemetryReasonCode.UNKNOWN,
                session_id=session.session_id,
                action=action,
                detail=detail,
            ),
            extra_identifiers={
                "caller_sid": caller.user_sid,
                "caller_pid": caller.pid,
            },
        )

    def _advance_fallback(
        self,
        *,
        session: _Session,
        caller: CallerIdentity,
        now_ms: int,
    ) -> dict[str, object]:
        state = FallbackState(
            current_step=session.fallback_current_step,
            last_step_change_ms=session.fallback_last_step_change_ms,
            step_change_history_ms=session.fallback_step_change_history_ms,
            stability_mode=session.fallback_stability_mode,
        )
        decision = self._fallback_controller.evaluate(state, now_ms)

        if decision.enter_stability_mode:
            session.fallback_stability_mode = True

        if not decision.should_advance:
            session.state = SessionState.DEGRADED if not session.fallback_stability_mode else SessionState.DEGRADED
            return {
                "should_advance": False,
                "next_step": session.fallback_current_step.value if session.fallback_current_step else None,
                "enter_stability_mode": session.fallback_stability_mode,
                "reason_code": decision.reason_code,
            }

        next_step = decision.next_step
        if next_step is None:
            session.state = SessionState.FAILED
            session.fallback_current_step = None
            session.fallback_last_step_change_ms = now_ms
            session.fallback_step_change_history_ms = session.fallback_step_change_history_ms + (now_ms,)
            self._record_fallback_event(
                session=session,
                caller=caller,
                ladder_step=FallbackStep.CONTROLLED_FAIL.value,
                telemetry_reason=TelemetryReasonCode.FALLBACK_CONTROLLED_FAIL,
                reason_code=decision.reason_code,
                now_ms=now_ms,
            )
            return {
                "should_advance": True,
                "next_step": FallbackStep.CONTROLLED_FAIL.value,
                "enter_stability_mode": session.fallback_stability_mode,
                "reason_code": decision.reason_code,
            }

        if next_step is FallbackStep.CONTROLLED_FAIL:
            session.fallback_current_step = next_step
            session.fallback_last_step_change_ms = now_ms
            session.fallback_step_change_history_ms = session.fallback_step_change_history_ms + (now_ms,)
            session.state = SessionState.FAILED
            self._record_fallback_event(
                session=session,
                caller=caller,
                ladder_step=next_step.value,
                telemetry_reason=TelemetryReasonCode.FALLBACK_CONTROLLED_FAIL,
                reason_code=decision.reason_code,
                now_ms=now_ms,
            )
            return {
                "should_advance": True,
                "next_step": next_step.value,
                "enter_stability_mode": session.fallback_stability_mode,
                "reason_code": decision.reason_code,
            }

        self._apply_deterministic_fallback(session=session, next_step=next_step, caller=caller)
        session.fallback_current_step = next_step
        session.fallback_last_step_change_ms = now_ms
        session.fallback_step_change_history_ms = session.fallback_step_change_history_ms + (now_ms,)
        session.state = SessionState.DEGRADED
        self._record_fallback_event(
            session=session,
            caller=caller,
            ladder_step=next_step.value,
            telemetry_reason=self._telemetry_reason_for_step(next_step),
            reason_code=decision.reason_code,
            now_ms=now_ms,
        )
        return {
            "should_advance": True,
            "next_step": next_step.value,
            "enter_stability_mode": session.fallback_stability_mode,
            "reason_code": decision.reason_code,
        }

    def _apply_deterministic_fallback(
        self,
        session: _Session,
        next_step: FallbackStep,
        *,
        caller: CallerIdentity | None = None,
    ) -> None:
        """Apply a deterministic fallback action to the highest-residency tensor."""

        tensor = self._select_tensor_for_fallback(session)
        if tensor is None:
            return

        if next_step is FallbackStep.KV_CONTEXT_REDUCTION:
            destination = self._fallback_destination(tensor.residency, preferred=ResidencyState.PINNED_RAM)
        elif next_step is FallbackStep.BATCH_REDUCTION:
            destination = self._fallback_destination(tensor.residency, preferred=ResidencyState.PAGEABLE_RAM)
        elif next_step is FallbackStep.PRECISION_REDUCTION:
            destination = self._fallback_destination(tensor.residency, preferred=ResidencyState.PAGEABLE_RAM)
        elif next_step is FallbackStep.SELECTIVE_CPU_OFFLOAD:
            destination = ResidencyState.CPU_ONLY
        else:
            return

        if destination != tensor.residency:
            self._migrate_tensor(
                session,
                tensor,
                destination,
                reason_code=next_step.value,
                telemetry_reason=self._telemetry_reason_for_step(next_step),
                telemetry_direction="fallback",
                telemetry_mode=next_step.value,
                caller=caller,
            )

    def _select_tensor_for_fallback(self, session: _Session) -> _TensorRecord | None:
        """Select the most memory-intensive tensor deterministically."""

        priority = {
            ResidencyState.VRAM: 0,
            ResidencyState.PINNED_RAM: 1,
            ResidencyState.PAGEABLE_RAM: 2,
            ResidencyState.CPU_ONLY: 3,
        }
        candidates = sorted(
            session.tensors.values(),
            key=lambda tensor: (priority.get(tensor.residency, 99), tensor.name),
        )
        return candidates[0] if candidates else None

    @staticmethod
    def _fallback_destination(
        current: ResidencyState,
        *,
        preferred: ResidencyState,
    ) -> ResidencyState:
        if current == preferred:
            if preferred == ResidencyState.PINNED_RAM:
                return ResidencyState.PAGEABLE_RAM
            if preferred == ResidencyState.PAGEABLE_RAM:
                return ResidencyState.CPU_ONLY
        return preferred

    def _record_fallback_event(
        self,
        *,
        session: _Session,
        caller: CallerIdentity,
        ladder_step: str,
        telemetry_reason: TelemetryReasonCode,
        reason_code: str,
        now_ms: int,
    ) -> None:
        core_event = FallbackEvent(
            session_id=session.session_id,
            step=ladder_step,
            reason_code=reason_code,
            timestamp_ms=now_ms,
        )
        session.fallback_events.append(core_event)
        self._telemetry.record_event(
            TelemetryFallbackEvent(
                reason_code=telemetry_reason,
                session_id=session.session_id,
                ladder_step=ladder_step,
                trigger="pressure_high",
                stabilization_mode=session.fallback_stability_mode,
            ),
            extra_identifiers={
                "caller_sid": caller.user_sid,
                "caller_pid": caller.pid,
            },
        )

    @staticmethod
    def _telemetry_reason_for_step(step: FallbackStep) -> TelemetryReasonCode:
        if step is FallbackStep.KV_CONTEXT_REDUCTION:
            return TelemetryReasonCode.FALLBACK_KV_REDUCTION
        if step is FallbackStep.BATCH_REDUCTION:
            return TelemetryReasonCode.FALLBACK_BATCH_REDUCTION
        if step is FallbackStep.PRECISION_REDUCTION:
            return TelemetryReasonCode.FALLBACK_PRECISION_REDUCTION
        if step is FallbackStep.SELECTIVE_CPU_OFFLOAD:
            return TelemetryReasonCode.FALLBACK_CPU_OFFLOAD
        return TelemetryReasonCode.FALLBACK_CONTROLLED_FAIL

    @staticmethod
    def _apply_residency_bytes(
        session: _Session,
        size_bytes: int,
        source: ResidencyState,
        destination: ResidencyState,
    ) -> None:
        def decrement(state: ResidencyState) -> None:
            if state is ResidencyState.VRAM:
                session.vram_used_bytes = max(0, session.vram_used_bytes - size_bytes)
            elif state is ResidencyState.PINNED_RAM:
                session.pinned_ram_used_bytes = max(0, session.pinned_ram_used_bytes - size_bytes)
            elif state is ResidencyState.PAGEABLE_RAM:
                session.pageable_ram_used_bytes = max(0, session.pageable_ram_used_bytes - size_bytes)
            elif state is ResidencyState.CPU_ONLY:
                session.cpu_only_used_bytes = max(0, session.cpu_only_used_bytes - size_bytes)

        def increment(state: ResidencyState) -> None:
            if state is ResidencyState.VRAM:
                session.vram_used_bytes += size_bytes
            elif state is ResidencyState.PINNED_RAM:
                session.pinned_ram_used_bytes += size_bytes
            elif state is ResidencyState.PAGEABLE_RAM:
                session.pageable_ram_used_bytes += size_bytes
            elif state is ResidencyState.CPU_ONLY:
                session.cpu_only_used_bytes += size_bytes

        decrement(source)
        increment(destination)

    @staticmethod
    def _estimate_transfer_latency_ms(size_bytes: int) -> float:
        """Return a deterministic latency estimate for telemetry."""

        return max(0.1, size_bytes / 100_000_000.0)

    def _resolve_runstep_mode(self) -> str:
        override = self._runstep_mode_override
        if isinstance(override, str) and override.strip():
            mode = override.strip().lower()
        else:
            mode = os.environ.get(RUNSTEP_MODE_ENV, DEFAULT_RUNSTEP_MODE).strip().lower()
            if self._env_flag_enabled(RUNSTEP_MODE_ENABLE_ENV):
                mode = "hardware"
        if mode in {"hardware", "cuda", "cuda_transfer_poc"}:
            return "hardware"
        if mode in {"auto", "hardware_preferred", "preferred"}:
            return "auto"
        return "simulation"

    def _execute_hardware_runstep(self, *, session: _Session, step_name: str) -> dict[str, object]:
        size_bytes = self._resolve_hardware_transfer_bytes(session)
        device_index = self._env_int(HARDWARE_DEVICE_INDEX_ENV, default=DEFAULT_DEVICE_INDEX, minimum=0)
        hold_ms = self._env_int(HARDWARE_HOLD_MS_ENV, default=DEFAULT_HARDWARE_HOLD_MS, minimum=0)

        if self._hardware_executor is None:
            return self._hardware_failure_result(
                code=ApiErrorCode.UNSUPPORTED_CAPABILITY,
                message="hardware-preferred RunStep could not load the CUDA runtime",
                session=session,
                step_name=step_name,
                size_bytes=size_bytes,
                device_index=device_index,
                hold_ms=hold_ms,
            )

        try:
            result = self._hardware_executor(
                size_bytes=size_bytes,
                device_index=device_index,
                hold_ms=hold_ms,
            )
        except TypeError:
            try:
                result = self._hardware_executor(size_bytes=size_bytes, device_index=device_index)
            except Exception as exc:  # pragma: no cover - defensive boundary
                return self._hardware_failure_result(
                    code=ApiErrorCode.INTERNAL,
                    message=f"hardware RunStep execution failed unexpectedly: {exc}",
                    session=session,
                    step_name=step_name,
                    size_bytes=size_bytes,
                    device_index=device_index,
                    hold_ms=hold_ms,
                )
        except Exception as exc:  # pragma: no cover - defensive boundary
            return self._hardware_failure_result(
                code=ApiErrorCode.INTERNAL,
                message=f"hardware RunStep execution failed unexpectedly: {exc}",
                session=session,
                step_name=step_name,
                size_bytes=size_bytes,
                device_index=device_index,
                hold_ms=hold_ms,
            )

        if not isinstance(result, dict):
            return self._hardware_failure_result(
                code=ApiErrorCode.INTERNAL,
                message="hardware RunStep must return a JSON object",
                session=session,
                step_name=step_name,
                size_bytes=size_bytes,
                device_index=device_index,
                hold_ms=hold_ms,
            )

        merged = dict(result)
        merged.setdefault("ok", False)
        merged["service_context"] = {
            "session_id": session.session_id,
            "step_name": step_name,
            "requested_transfer_bytes": size_bytes,
            "device_index": device_index,
            "hold_ms": hold_ms,
        }
        if not bool(merged.get("ok")) and "error" not in merged:
            merged["error"] = {
                "code": ApiErrorCode.INTERNAL.value,
                "message": "hardware RunStep reported failure without structured error details",
            }
        return merged

    def _resolve_hardware_transfer_bytes(self, session: _Session) -> int:
        configured = self._env_int(HARDWARE_TRANSFER_BYTES_ENV, default=0, minimum=0)
        if configured > 0:
            return configured
        if session.tensors:
            return max(DEFAULT_TRANSFER_BYTES, max(item.size_bytes for item in session.tensors.values()))
        return DEFAULT_TRANSFER_BYTES

    def _execute_inference_runstep(
        self,
        *,
        session: _Session,
        step_name: str,
        prompt: str,
        max_tokens: int | None,
        temperature: float | None,
        system_prompt: str | None,
        backend_options: Mapping[str, Any] | None,
    ) -> dict[str, object]:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "prompt must be a non-empty string when provided")
        if max_tokens is not None and max_tokens <= 0:
            raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "max_tokens must be positive when provided")
        if temperature is not None and temperature < 0:
            raise ApiError(ApiErrorCode.INVALID_ARGUMENT, "temperature must be non-negative when provided")
        if session.resolved_model_name is None:
            raise ApiError(ApiErrorCode.INVALID_STATE, "session has no resolved model binding")

        runtime = self._inference_runtime_factory(session.inference_backend)
        try:
            result = self._call_runtime_method(
                runtime,
                "generate",
                session.resolved_model_name,
                prompt=prompt,
                step_name=step_name,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=system_prompt,
                backend_options=backend_options,
            )
        except ApiError as exc:
            return {
                "ok": False,
                "backend": session.inference_backend,
                "model_name": session.resolved_model_name,
                "error": {
                    "code": exc.code.value,
                    "message": str(exc),
                },
            }
        except Exception as exc:  # pragma: no cover - defensive runtime boundary
            return {
                "ok": False,
                "backend": session.inference_backend,
                "model_name": session.resolved_model_name,
                "error": {
                    "code": ApiErrorCode.INTERNAL.value,
                    "message": f"inference backend execution failed unexpectedly: {exc}",
                },
            }

        if not isinstance(result, dict):
            return {
                "ok": False,
                "backend": session.inference_backend,
                "model_name": session.resolved_model_name,
                "error": {
                    "code": ApiErrorCode.INTERNAL.value,
                    "message": "inference backend must return a JSON object",
                },
            }

        normalized = dict(result)
        normalized.setdefault("ok", True)
        normalized.setdefault("backend", session.inference_backend)
        normalized.setdefault("model_name", session.resolved_model_name)
        return normalized

    @staticmethod
    def _hardware_failure_result(
        *,
        code: ApiErrorCode,
        message: str,
        session: _Session,
        step_name: str,
        size_bytes: int,
        device_index: int,
        hold_ms: int,
    ) -> dict[str, object]:
        return {
            "ok": False,
            "error": {
                "code": code.value,
                "message": message,
            },
            "fallback_to_simulation": True,
            "service_context": {
                "session_id": session.session_id,
                "step_name": step_name,
                "requested_transfer_bytes": size_bytes,
                "device_index": device_index,
                "hold_ms": hold_ms,
            },
        }

    @staticmethod
    def _env_flag_enabled(name: str) -> bool:
        value = os.environ.get(name, "").strip().lower()
        return value in {"1", "true", "yes", "on"}

    @staticmethod
    def _env_int(name: str, *, default: int, minimum: int = 0) -> int:
        raw = os.environ.get(name)
        if raw is None or not raw.strip():
            return default
        try:
            value = int(raw.strip())
        except ValueError:
            return default
        if value < minimum:
            return default
        return value

    def _compute_pressure_level(self, session: _Session) -> float:
        if session.vram_budget_bytes <= 0:
            return 1.0
        level = session.vram_used_bytes / session.vram_budget_bytes
        return max(0.0, min(1.0, level))

    @staticmethod
    def _derive_primary_tier(session: _Session) -> MemoryTier:
        if any(tensor.residency is ResidencyState.VRAM for tensor in session.tensors.values()):
            return MemoryTier.HOT
        if any(tensor.residency is ResidencyState.PINNED_RAM for tensor in session.tensors.values()):
            return MemoryTier.WARM
        return MemoryTier.COLD

    @staticmethod
    def _now_ms() -> int:
        return time_ns() // 1_000_000

    @staticmethod
    def _is_valid_caller_identity(caller_identity: CallerIdentity | None) -> bool:
        if caller_identity is None:
            return False
        try:
            return (
                bool(caller_identity.user_sid.strip())
                and isinstance(caller_identity.pid, int)
                and not isinstance(caller_identity.pid, bool)
                and caller_identity.pid > 0
            )
        except AttributeError:
            return False

    @staticmethod
    def _call_runtime_method(runtime: Any, method_name: str, *args: Any, **kwargs: Any) -> Any:
        method = getattr(runtime, method_name)
        try:
            params = signature(method).parameters
        except (TypeError, ValueError):
            return method(*args, **kwargs)

        if any(param.kind == Parameter.VAR_KEYWORD for param in params.values()):
            return method(*args, **kwargs)

        filtered_kwargs = {key: value for key, value in kwargs.items() if key in params}
        return method(*args, **filtered_kwargs)

    @staticmethod
    def _build_inference_metadata(
        binding_metadata: Any,
        *,
        requested_model_name: str,
        runtime_profile: str,
        model_size_billion: float | None,
        model_size_label: str | None,
        backend_options: Mapping[str, Any],
    ) -> dict[str, Any]:
        metadata = dict(binding_metadata) if isinstance(binding_metadata, Mapping) else {}
        metadata["requested_model_name"] = requested_model_name
        metadata["runtime_profile"] = runtime_profile
        metadata["backend_options"] = dict(backend_options)
        if model_size_billion is not None:
            metadata["model_size_billion"] = model_size_billion
        if model_size_label is not None:
            metadata["model_size_label"] = model_size_label
        return metadata

    @staticmethod
    def _resolve_effective_backend_options(
        *,
        session: _Session,
        runtime_profile_override: str | None,
        runtime_backend_options_override: Mapping[str, Any] | None,
        backend_options: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        stored_options = session.inference_metadata.get("backend_options")
        if not isinstance(stored_options, Mapping):
            stored_options = {}
        profile_options: Mapping[str, Any] | None = None
        if runtime_profile_override is not None:
            resolved_model_name = session.resolved_model_name
            if not isinstance(resolved_model_name, str) or not resolved_model_name.strip():
                raise ApiError(ApiErrorCode.INVALID_STATE, "session has no resolved model binding")
            override_tuning = resolve_runtime_tuning(
                resolved_model_name,
                runtime_profile=runtime_profile_override,
            )
            profile_options = override_tuning.backend_options
        return merge_backend_options(
            stored_options,
            profile_options,
            runtime_backend_options_override,
            backend_options,
        )


__all__ = ["AstraWeaveService"]
