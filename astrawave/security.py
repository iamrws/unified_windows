"""Local-only trust boundary helpers for AstraWeave.

This module is intentionally framework-agnostic. It provides pure data types
and a small in-memory guard that service code can use to enforce caller
identity, authorization, rate limits, and concurrent-session caps.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from time import monotonic
from typing import Callable, Deque, Dict, FrozenSet, Optional, Tuple

from .errors import ApiError, ApiErrorCode

CallerKey = Tuple[str, int]
Clock = Callable[[], float]

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
    allowed_cross_user_sids: FrozenSet[str] = field(default_factory=frozenset)
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
    reason: Optional[SecurityDenyReason] = None
    message: str = ""
    retry_after_seconds: Optional[int] = None

    def to_api_error(self) -> Optional[ApiError]:
        """Convert a deny decision into a stable API error object."""

        if self.allowed:
            return None
        return ApiError(self.error_code, self.message or self.error_code.value)


def decision_allowed(message: str = "") -> SecurityDecision:
    """Create an allow decision."""

    return SecurityDecision(True, ApiErrorCode.OK, None, message)


def decision_denied(
    reason: SecurityDenyReason,
    message: str,
    *,
    retry_after_seconds: Optional[int] = None,
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
        self._create_session_attempts: Dict[CallerKey, Deque[float]] = {}
        self._active_sessions: Dict[CallerKey, int] = {}

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

    def can_create_session(self, caller: CallerIdentity, *, now: Optional[float] = None) -> SecurityDecision:
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

    def admit_create_session(self, caller: CallerIdentity, *, now: Optional[float] = None) -> SecurityDecision:
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

    def _resolve_now(self, now: Optional[float]) -> float:
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
    "CallerIdentity",
    "Clock",
    "CREATE_SESSION_LIMIT_PER_MINUTE",
    "ERROR_CODE_BY_DENY_REASON",
    "MAX_CONCURRENT_SESSIONS_PER_CALLER",
    "RATE_WINDOW_SECONDS",
    "SecurityDecision",
    "SecurityDenyReason",
    "SecurityGuard",
    "SecurityPolicy",
    "decision_allowed",
    "decision_denied",
    "is_valid_caller_identity",
]
