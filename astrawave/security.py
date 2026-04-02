"""Local-only trust boundary helpers for AstraWeave.

This module is intentionally framework-agnostic. It provides pure data types
and a small in-memory guard that service code can use to enforce caller
identity, authorization, rate limits, and concurrent-session caps.
"""

from __future__ import annotations

from collections import deque
import errno
from dataclasses import dataclass, field
from enum import Enum
import os
from threading import RLock
from time import monotonic
from collections.abc import Callable
from typing import Deque

# M27 fix: guard Windows-only imports
if os.name == "nt":
    import ctypes
    from ctypes import wintypes
else:
    ctypes = None  # type: ignore[assignment]
    wintypes = None  # type: ignore[assignment]

from .errors import ApiError, ApiErrorCode

CallerKey = tuple[str, int]
Clock = Callable[[], float]

PidLookup = Callable[[int], bool]
SidLookup = Callable[[int], str | None]


if ctypes is not None:
    class _SidAndAttributes(ctypes.Structure):
        _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", wintypes.DWORD)]

    class _TokenUser(ctypes.Structure):
        _fields_ = [("User", _SidAndAttributes)]
else:
    _SidAndAttributes = None  # type: ignore[assignment,misc]
    _TokenUser = None  # type: ignore[assignment,misc]

CREATE_SESSION_LIMIT_PER_MINUTE = 6
MAX_CONCURRENT_SESSIONS_PER_CALLER = 8
RATE_WINDOW_SECONDS = 60.0


class SecurityDenyReason(str, Enum):
    """Deterministic denial reasons for security decisions."""

    INVALID_CALLER = "INVALID_CALLER"
    UNKNOWN_CALLER = "UNKNOWN_CALLER"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    MAX_CONCURRENT_SESSIONS = "MAX_CONCURRENT_SESSIONS"


ERROR_CODE_BY_DENY_REASON: dict[SecurityDenyReason, ApiErrorCode] = {
    SecurityDenyReason.INVALID_CALLER: ApiErrorCode.INVALID_ARGUMENT,
    SecurityDenyReason.UNKNOWN_CALLER: ApiErrorCode.AUTH_DENIED,
    SecurityDenyReason.RATE_LIMIT_EXCEEDED: ApiErrorCode.RATE_LIMITED,
    SecurityDenyReason.MAX_CONCURRENT_SESSIONS: ApiErrorCode.RESOURCE_EXHAUSTED,
}


@dataclass(frozen=True, slots=True)
class CallerIdentity:
    """Identity tuple used by the trust-boundary helpers.

    The service authorizes callers by user SID and tracks usage by the full
    caller key (SID + PID) so same-user processes remain isolated for rate
    limiting and session-count accounting.
    """

    user_sid: str
    pid: int

    def __post_init__(self) -> None:
        if not self.user_sid or not self.user_sid.strip():
            raise ValueError("user_sid must be a non-empty string")
        if not isinstance(self.pid, int) or isinstance(self.pid, bool) or self.pid <= 0:
            raise ValueError("pid must be a positive integer")

    @property
    def key(self) -> CallerKey:
        return (self.user_sid, self.pid)


@dataclass(frozen=True, slots=True)
class SecurityPolicy:
    """Immutable authorization policy for the local service."""

    service_owner_sid: str
    allowed_cross_user_sids: frozenset[str] = field(default_factory=frozenset)
    create_session_limit_per_minute: int = CREATE_SESSION_LIMIT_PER_MINUTE
    max_concurrent_sessions_per_caller: int = MAX_CONCURRENT_SESSIONS_PER_CALLER
    rate_window_seconds: float = RATE_WINDOW_SECONDS

    def __post_init__(self) -> None:
        if not self.service_owner_sid or not self.service_owner_sid.strip():
            raise ValueError("service_owner_sid must be a non-empty string")
        if self.create_session_limit_per_minute <= 0:
            raise ValueError("create_session_limit_per_minute must be positive")
        if self.max_concurrent_sessions_per_caller <= 0:
            raise ValueError("max_concurrent_sessions_per_caller must be positive")
        if self.rate_window_seconds <= 0:
            raise ValueError("rate_window_seconds must be positive")
        normalized = frozenset(
            sid.strip() for sid in self.allowed_cross_user_sids if sid and sid.strip()
        )
        object.__setattr__(self, "allowed_cross_user_sids", normalized)


@dataclass(frozen=True, slots=True)
class SecurityDecision:
    """Result of an authorization or admission check."""

    allowed: bool
    error_code: ApiErrorCode = ApiErrorCode.OK
    reason: SecurityDenyReason | None = None
    message: str = ""
    retry_after_seconds: int | None = None

    def to_api_error(self) -> ApiError | None:
        """Convert a deny decision into a stable API error object."""

        if self.allowed:
            return None
        result = ApiError(self.error_code, self.message or self.error_code.value)
        assert result is not None, "deny decision must produce a non-None ApiError"
        return result


def decision_allowed(message: str = "") -> SecurityDecision:
    """Create an allow decision."""

    return SecurityDecision(True, ApiErrorCode.OK, None, message)


def decision_denied(
    reason: SecurityDenyReason,
    message: str,
    *,
    retry_after_seconds: int | None = None,
) -> SecurityDecision:
    """Create a deterministic deny decision for a given reason."""

    return SecurityDecision(
        allowed=False,
        error_code=ERROR_CODE_BY_DENY_REASON[reason],
        reason=reason,
        message=message,
        retry_after_seconds=retry_after_seconds,
    )


def is_valid_caller_identity(caller: CallerIdentity) -> bool:
    """Return `True` when the caller identity is structurally valid."""

    try:
        return (
            bool(caller.user_sid.strip())
            and isinstance(caller.pid, int)
            and not isinstance(caller.pid, bool)
            and caller.pid > 0
        )
    except AttributeError:
        return False


def is_plausible_user_sid(user_sid: str) -> bool:
    """Return `True` when the caller SID is structurally safe to trust-check."""

    if not isinstance(user_sid, str):
        return False
    normalized = user_sid.strip()
    if not normalized or len(normalized) > 256:
        return False
    if any(ord(ch) < 32 for ch in normalized):
        return False
    if any(ch.isspace() for ch in normalized):
        return False
    return normalized.startswith("S-")


def process_exists(pid: int) -> bool:
    """Return `True` when the PID currently maps to a live local process."""

    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        process_handle = kernel32.OpenProcess(0x1000, False, pid)
        if process_handle:
            kernel32.CloseHandle(process_handle)
            return True
        return ctypes.get_last_error() == 5
    try:
        os.kill(pid, 0)
    except OSError as exc:
        return exc.errno == errno.EPERM
    return True


def resolve_process_user_sid(pid: int) -> str | None:
    """Return the Windows SID string for a process owner when available."""

    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return None
    if os.name != "nt":
        return None

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)

    # M26 fix: removed dead wintypes.HANDLE() allocation
    token_handle = wintypes.HANDLE()
    sid_string = wintypes.LPWSTR()
    process_handle = kernel32.OpenProcess(0x1000, False, pid)
    if not process_handle:
        return None

    try:
        if not advapi32.OpenProcessToken(process_handle, 0x0008, ctypes.byref(token_handle)):
            return None
        try:
            required = wintypes.DWORD(0)
            advapi32.GetTokenInformation(token_handle, 1, None, 0, ctypes.byref(required))
            if required.value <= 0:
                return None

            buffer = ctypes.create_string_buffer(required.value)
            if not advapi32.GetTokenInformation(token_handle, 1, buffer, required.value, ctypes.byref(required)):
                return None

            token_user = _TokenUser.from_buffer_copy(buffer)
            sid_ptr = ctypes.c_void_p(token_user.User.Sid)
            if not sid_ptr.value:
                return None
            if not advapi32.ConvertSidToStringSidW(sid_ptr, ctypes.byref(sid_string)):
                return None
            try:
                return str(sid_string.value) if sid_string.value else None
            finally:
                ctypes.windll.kernel32.LocalFree(sid_string)
        finally:
            kernel32.CloseHandle(token_handle)
    finally:
        kernel32.CloseHandle(process_handle)


def resolve_current_user_sid() -> str | None:
    """Return the current process owner SID when the platform can provide it."""

    return resolve_process_user_sid(os.getpid())


def _attest_caller_via_handle(caller: CallerIdentity) -> SecurityDecision:
    """Windows-specific attestation that holds a single process handle.

    By opening the process handle once and using it for both the existence
    check and the SID resolution, the TOCTOU window between the two
    lookups is eliminated on Windows.  The handle keeps a reference to
    the kernel process object, so PID reuse cannot silently substitute a
    different process between the two checks.
    """
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)

    # PROCESS_QUERY_LIMITED_INFORMATION (0x1000) is sufficient for
    # OpenProcessToken and keeps the required privilege minimal.
    process_handle = kernel32.OpenProcess(0x1000, False, caller.pid)
    if not process_handle:
        # Access denied (error 5) means the process exists but we
        # cannot open it — treat that as "active but unverifiable SID".
        if ctypes.get_last_error() == 5:
            return decision_allowed(
                "caller PID verified (access denied); OS SID attestation unavailable"
            )
        return decision_denied(
            SecurityDenyReason.UNKNOWN_CALLER,
            "caller PID is not active on this host",
        )

    try:
        # --- resolve SID using the same handle ---
        token_handle = wintypes.HANDLE()
        if not advapi32.OpenProcessToken(
            process_handle, 0x0008, ctypes.byref(token_handle)
        ):
            # Could not open the token — PID is alive but SID unknown.
            return decision_allowed(
                "caller PID verified; OS SID attestation unavailable"
            )
        try:
            required = wintypes.DWORD(0)
            advapi32.GetTokenInformation(
                token_handle, 1, None, 0, ctypes.byref(required)
            )
            if required.value <= 0:
                return decision_allowed(
                    "caller PID verified; OS SID attestation unavailable"
                )

            buffer = ctypes.create_string_buffer(required.value)
            if not advapi32.GetTokenInformation(
                token_handle, 1, buffer, required.value, ctypes.byref(required)
            ):
                return decision_allowed(
                    "caller PID verified; OS SID attestation unavailable"
                )

            token_user = _TokenUser.from_buffer_copy(buffer)
            sid_ptr = ctypes.c_void_p(token_user.User.Sid)
            sid_string = wintypes.LPWSTR()
            if not sid_ptr.value or not advapi32.ConvertSidToStringSidW(
                sid_ptr, ctypes.byref(sid_string)
            ):
                return decision_allowed(
                    "caller PID verified; OS SID attestation unavailable"
                )

            try:
                resolved_sid = (
                    str(sid_string.value) if sid_string.value else None
                )
            finally:
                ctypes.windll.kernel32.LocalFree(sid_string)

            if resolved_sid is None:
                return decision_allowed(
                    "caller PID verified; OS SID attestation unavailable"
                )

            if resolved_sid.strip() != caller.user_sid.strip():
                return decision_denied(
                    SecurityDenyReason.UNKNOWN_CALLER,
                    "caller SID does not match the active process owner",
                )

            return decision_allowed(
                "caller attested by PID and process owner SID"
            )
        finally:
            kernel32.CloseHandle(token_handle)
    finally:
        kernel32.CloseHandle(process_handle)


def attest_caller_identity(
    caller: CallerIdentity,
    *,
    pid_lookup: PidLookup | None = None,
    sid_lookup: SidLookup | None = None,
) -> SecurityDecision:
    """Attest the caller against the live process table when possible.

    Security note — PID-reuse TOCTOU (FINDING-009)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    On all operating systems, PIDs can be recycled by the kernel once a
    process exits.  Between the moment this function checks that a PID is
    alive and the moment it resolves the owning SID, the original process
    could exit and its PID could be reassigned to an unrelated process.

    On Windows this risk is mitigated: when no custom ``pid_lookup`` /
    ``sid_lookup`` overrides are supplied, the function delegates to
    ``_attest_caller_via_handle`` which opens a **single** process handle
    and reuses it for both checks.  Because a kernel handle holds a
    reference to the process object, PID reuse cannot substitute a
    different process for the duration of the call.

    When custom lookup callables *are* supplied (primarily for testing),
    the two-step PID-then-SID flow is used, which is subject to the
    TOCTOU window.

    Recommended long-term mitigation: callers should open a process
    handle at session-creation time, hold it for the session lifetime,
    and pass it into future attestation calls.  This completely
    eliminates PID-reuse risk for the session duration.
    """

    if not is_valid_caller_identity(caller) or not is_plausible_user_sid(caller.user_sid):
        return decision_denied(
            SecurityDenyReason.INVALID_CALLER,
            "caller identity is malformed",
        )

    # FINDING-009 improvement: when running on Windows with default
    # lookups, use the single-handle path to eliminate the TOCTOU window
    # between the PID existence check and the SID resolution.
    if os.name == "nt" and pid_lookup is None and sid_lookup is None:
        return _attest_caller_via_handle(caller)

    pid_lookup = pid_lookup or process_exists
    sid_lookup = sid_lookup or resolve_process_user_sid

    try:
        pid_is_active = pid_lookup(caller.pid)
    except Exception:
        pid_is_active = False
    if not pid_is_active:
        return decision_denied(
            SecurityDenyReason.UNKNOWN_CALLER,
            "caller PID is not active on this host",
        )

    try:
        resolved_sid = sid_lookup(caller.pid)
    except Exception:
        resolved_sid = None

    if resolved_sid is not None and resolved_sid.strip() != caller.user_sid.strip():
        return decision_denied(
            SecurityDenyReason.UNKNOWN_CALLER,
            "caller SID does not match the active process owner",
        )

    if resolved_sid is not None:
        return decision_allowed("caller attested by PID and process owner SID")
    return decision_allowed("caller PID verified; OS SID attestation unavailable")


class SecurityGuard:
    """In-memory authorization and abuse-control helper.

    The guard enforces:
    - same-user default allow
    - cross-user explicit allowlist
    - rate limiting for CreateSession attempts
    - maximum concurrent sessions per caller

    It is safe to use from service code and easy to unit test thanks to the
    injectable monotonic clock.
    """

    def __init__(
        self,
        policy: SecurityPolicy,
        *,
        clock: Clock = monotonic,
    ) -> None:
        self._policy = policy
        self._clock = clock
        self._lock = RLock()
        self._create_session_attempts: dict[CallerKey, Deque[float]] = {}
        self._active_sessions: dict[CallerKey, int] = {}

    @property
    def policy(self) -> SecurityPolicy:
        """Return the current security policy."""

        return self._policy

    def authorize_caller(self, caller: CallerIdentity) -> SecurityDecision:
        """Check whether a caller is allowed to use the local service."""

        if not is_valid_caller_identity(caller):
            return decision_denied(
                SecurityDenyReason.INVALID_CALLER,
                "caller identity is malformed",
            )

        if caller.user_sid == self._policy.service_owner_sid:
            return decision_allowed("same-user caller accepted")

        if caller.user_sid in self._policy.allowed_cross_user_sids:
            return decision_allowed("cross-user caller accepted by policy")

        return decision_denied(
            SecurityDenyReason.UNKNOWN_CALLER,
            "caller is not authorized for this local service",
        )

    def can_create_session(self, caller: CallerIdentity, *, now: float | None = None) -> SecurityDecision:
        """Check admission for `CreateSession` without mutating state."""

        auth = self.authorize_caller(caller)
        if not auth.allowed:
            return auth

        with self._lock:
            moment = self._resolve_now(now)
            attempts = self._get_attempt_bucket(caller.key)
            self._prune_old_attempts(attempts, moment)

            if len(attempts) >= self._policy.create_session_limit_per_minute:
                retry_after = self._retry_after_seconds(attempts, moment)
                return decision_denied(
                    SecurityDenyReason.RATE_LIMIT_EXCEEDED,
                    "CreateSession rate limit exceeded",
                    retry_after_seconds=retry_after,
                )

            active_sessions = self._active_sessions.get(caller.key, 0)
            if active_sessions >= self._policy.max_concurrent_sessions_per_caller:
                return decision_denied(
                    SecurityDenyReason.MAX_CONCURRENT_SESSIONS,
                    "maximum concurrent sessions reached for caller",
                )

            return decision_allowed("caller admitted for CreateSession")

    def admit_create_session(self, caller: CallerIdentity, *, now: float | None = None) -> SecurityDecision:
        """Atomically approve and record a `CreateSession` attempt.

        Call this only when the service is ready to create a session. On
        success, the caller is counted toward both the rolling rate limit and
        the concurrent-session cap. Denials caused by concurrent-session
        exhaustion still count as attempts so repeated hammering is not free.
        """

        auth = self.authorize_caller(caller)
        if not auth.allowed:
            return auth

        with self._lock:
            moment = self._resolve_now(now)
            attempts = self._get_attempt_bucket(caller.key)
            self._prune_old_attempts(attempts, moment)

            if len(attempts) >= self._policy.create_session_limit_per_minute:
                retry_after = self._retry_after_seconds(attempts, moment)
                return decision_denied(
                    SecurityDenyReason.RATE_LIMIT_EXCEEDED,
                    "CreateSession rate limit exceeded",
                    retry_after_seconds=retry_after,
                )

            attempts.append(moment)
            active_sessions = self._active_sessions.get(caller.key, 0)
            if active_sessions >= self._policy.max_concurrent_sessions_per_caller:
                return decision_denied(
                    SecurityDenyReason.MAX_CONCURRENT_SESSIONS,
                    "maximum concurrent sessions reached for caller",
                )

            self._active_sessions[caller.key] = active_sessions + 1
            return decision_allowed("CreateSession admitted")

    def release_session(self, caller: CallerIdentity) -> None:
        """Release one active session slot for a caller.

        The operation is intentionally idempotent. Releasing when no active
        session is tracked is a no-op.
        """

        if not is_valid_caller_identity(caller):
            return

        with self._lock:
            active_sessions = self._active_sessions.get(caller.key, 0)
            if active_sessions <= 1:
                self._active_sessions.pop(caller.key, None)
            else:
                self._active_sessions[caller.key] = active_sessions - 1

    def active_session_count(self, caller: CallerIdentity) -> int:
        """Return the currently tracked active sessions for a caller."""

        if not is_valid_caller_identity(caller):
            return 0
        with self._lock:
            return self._active_sessions.get(caller.key, 0)

    def snapshot(self) -> dict[CallerKey, dict[str, int]]:
        """Return a test-friendly snapshot of internal counters."""

        with self._lock:
            return {
                caller_key: {
                    "active_sessions": self._active_sessions.get(caller_key, 0),
                    "recent_create_session_attempts": len(
                        self._create_session_attempts.get(caller_key, deque())
                    ),
                }
                for caller_key in set(self._create_session_attempts) | set(self._active_sessions)
            }

    def _resolve_now(self, now: float | None) -> float:
        return self._clock() if now is None else now

    def _get_attempt_bucket(self, caller_key: CallerKey) -> Deque[float]:
        bucket = self._create_session_attempts.get(caller_key)
        if bucket is None:
            bucket = deque()
            self._create_session_attempts[caller_key] = bucket
        return bucket

    def _prune_old_attempts(self, attempts: Deque[float], now: float) -> None:
        cutoff = now - self._policy.rate_window_seconds
        while attempts and attempts[0] < cutoff:
            attempts.popleft()

    def _retry_after_seconds(self, attempts: Deque[float], now: float) -> int:
        if not attempts:
            return int(self._policy.rate_window_seconds)
        oldest = attempts[0]
        remaining = self._policy.rate_window_seconds - (now - oldest)
        if remaining <= 0:
            return 0
        return int(remaining) + (0 if remaining.is_integer() else 1)


__all__ = [
    "attest_caller_identity",
    "CallerIdentity",
    "Clock",
    "CREATE_SESSION_LIMIT_PER_MINUTE",
    "ERROR_CODE_BY_DENY_REASON",
    "MAX_CONCURRENT_SESSIONS_PER_CALLER",
    "PidLookup",
    "RATE_WINDOW_SECONDS",
    "SecurityDecision",
    "SecurityDenyReason",
    "SecurityGuard",
    "SecurityPolicy",
    "SidLookup",
    "decision_allowed",
    "decision_denied",
    "is_valid_caller_identity",
    "is_plausible_user_sid",
    "process_exists",
    "resolve_current_user_sid",
    "resolve_process_user_sid",
]
