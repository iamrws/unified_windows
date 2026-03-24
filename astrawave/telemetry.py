"""Telemetry pipeline primitives for AstraWeave.

This module keeps telemetry local by default, redacts sensitive values before
persisting them, and only allows export bundles after explicit opt-in.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import re
import uuid
from typing import Any, Callable, ClassVar, Mapping

from .errors import ApiError, ApiErrorCode


def utcnow() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)


def new_correlation_id(prefix: str = "tel") -> str:
    """Generate a stable-format correlation id for telemetry events."""

    return f"{prefix}-{uuid.uuid4()}"


def hash_identifier(value: object, salt: str = "") -> str:
    """Hash an identifier for persisted telemetry records.

    The output is deterministic for the same input and salt, but the raw value
    is not preserved in persisted telemetry.
    """

    digest = hashlib.sha256(f"{salt}|{value!s}".encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _coerce_datetime(value: datetime | None) -> datetime:
    if value is None:
        return utcnow()
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _coerce_reason_code(value: str | TelemetryReasonCode) -> TelemetryReasonCode:
    if isinstance(value, TelemetryReasonCode):
        return value
    try:
        return TelemetryReasonCode(value)
    except ValueError as exc:  # pragma: no cover - defensive branch
        raise ApiError(
            ApiErrorCode.INVALID_ARGUMENT,
            f"Unknown telemetry reason code: {value}",
        ) from exc


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(
        token in lowered
        for token in (
            "prompt",
            "completion",
            "output",
            "response",
            "secret",
            "token",
            "password",
            "passphrase",
            "credential",
            "authorization",
            "bearer",
            "cookie",
            "api_key",
            "apikey",
            "file_contents",
            "content",
        )
    )


def _is_identifier_key(key: str) -> bool:
    lowered = key.lower()
    return lowered in {
        "session_id",
        "caller_id",
        "caller_sid",
        "caller_pid",
        "process_id",
        "pid",
    } or lowered.endswith("_id")


_SECRET_PATTERNS = [
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-+=/]+\b"),
    re.compile(r"(?i)\b(?:api[_-]?key|secret|token|password|passphrase)\b\s*[:=]\s*\S+"),
]


def redact_text(text: str) -> str:
    """Redact obvious secret material from free-form text."""

    redacted = text
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("<redacted>", redacted)
    return redacted


def redact_value(
    value: Any,
    *,
    salt: str = "",
    allow_debug_identifiers: bool = False,
) -> Any:
    """Redact or hash a value recursively."""

    if value is None:
        return None
    if isinstance(value, Mapping):
        return redact_mapping(
            value,
            salt=salt,
            allow_debug_identifiers=allow_debug_identifiers,
        )
    if isinstance(value, (list, tuple)):
        return [
            redact_value(
                item,
                salt=salt,
                allow_debug_identifiers=allow_debug_identifiers,
            )
            for item in value
        ]
    if isinstance(value, str):
        return redact_text(value)
    return value


def redact_mapping(
    data: Mapping[str, Any],
    *,
    salt: str = "",
    allow_debug_identifiers: bool = False,
) -> dict[str, Any]:
    """Redact a mapping while hashing identifiers for persisted records."""

    redacted: dict[str, Any] = {}
    for key, value in data.items():
        if _is_sensitive_key(key):
            redacted[key] = "<redacted>"
            continue
        if _is_identifier_key(key) and not allow_debug_identifiers:
            redacted[key] = hash_identifier(value, salt=salt)
            continue
        redacted[key] = redact_value(
            value,
            salt=salt,
            allow_debug_identifiers=allow_debug_identifiers,
        )
    return redacted


class TelemetryEventType(str, Enum):
    """Structured telemetry event kinds."""

    CUSTOM = "custom"
    RESIDENCY_CHANGE = "residency_change"
    TRANSFER = "transfer_event"
    FALLBACK = "fallback_event"
    PRESSURE = "pressure_snapshot"
    SECURITY = "security_event"
    POLICY = "policy_event"


class TelemetryReasonCode(str, Enum):
    """Stable reason codes used by telemetry events and exports."""

    UNKNOWN = "AW_TEL_REASON_UNKNOWN"
    RESIDENCY_PROMOTED = "AW_TEL_REASON_RESIDENCY_PROMOTED"
    RESIDENCY_DEMOTED = "AW_TEL_REASON_RESIDENCY_DEMOTED"
    TRANSFER_PREFETCH = "AW_TEL_REASON_TRANSFER_PREFETCH"
    TRANSFER_EVICTION = "AW_TEL_REASON_TRANSFER_EVICTION"
    FALLBACK_KV_REDUCTION = "AW_TEL_REASON_FALLBACK_KV_REDUCTION"
    FALLBACK_BATCH_REDUCTION = "AW_TEL_REASON_FALLBACK_BATCH_REDUCTION"
    FALLBACK_PRECISION_REDUCTION = "AW_TEL_REASON_FALLBACK_PRECISION_REDUCTION"
    FALLBACK_CPU_OFFLOAD = "AW_TEL_REASON_FALLBACK_CPU_OFFLOAD"
    FALLBACK_CONTROLLED_FAIL = "AW_TEL_REASON_FALLBACK_CONTROLLED_FAIL"
    PRESSURE_RISE = "AW_TEL_REASON_PRESSURE_RISE"
    PRESSURE_RELIEF = "AW_TEL_REASON_PRESSURE_RELIEF"
    SECURITY_DENY = "AW_TEL_REASON_SECURITY_DENY"
    POLICY_CHANGED = "AW_TEL_REASON_POLICY_CHANGED"
    EXPORT_BLOCKED = "AW_TEL_REASON_EXPORT_BLOCKED"
    RETENTION_CLEANUP = "AW_TEL_REASON_RETENTION_CLEANUP"


class TelemetryRecordClass(str, Enum):
    """Record classes used by retention rules."""

    EVENT = "event"
    METRIC = "metric"
    CRASH = "crash"
    BUNDLE = "bundle"


@dataclass(frozen=True, slots=True, kw_only=True)
class TelemetryEvent:
    """Base telemetry event with a correlation id and stable reason code."""

    reason_code: TelemetryReasonCode | str
    correlation_id: str = field(default_factory=new_correlation_id)
    session_id: str | None = None
    timestamp: datetime | None = None

    event_type: ClassVar[TelemetryEventType] = TelemetryEventType.CUSTOM
    record_class: ClassVar[TelemetryRecordClass] = TelemetryRecordClass.EVENT

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason_code", _coerce_reason_code(self.reason_code))
        object.__setattr__(self, "timestamp", _coerce_datetime(self.timestamp))
        if not self.correlation_id.strip():
            raise ApiError(
                ApiErrorCode.INVALID_ARGUMENT,
                "Telemetry events require a non-empty correlation id.",
            )
        if self.session_id is not None and not self.session_id.strip():
            raise ApiError(
                ApiErrorCode.INVALID_ARGUMENT,
                "Telemetry session ids cannot be empty.",
            )

    def structured_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly structured event dictionary."""

        structured = asdict(self)
        structured["event_type"] = self.event_type.value
        structured["record_class"] = self.record_class.value
        structured["reason_code"] = self.reason_code.value
        structured["timestamp"] = self.timestamp.isoformat()
        return structured


@dataclass(frozen=True, slots=True, kw_only=True)
class ResidencyChangeEvent(TelemetryEvent):
    event_type: ClassVar[TelemetryEventType] = TelemetryEventType.RESIDENCY_CHANGE
    tensor_class: str
    from_state: str
    to_state: str


@dataclass(frozen=True, slots=True, kw_only=True)
class TransferEvent(TelemetryEvent):
    event_type: ClassVar[TelemetryEventType] = TelemetryEventType.TRANSFER
    bytes_moved: int
    direction: str
    latency_ms: float
    mode: str


@dataclass(frozen=True, slots=True, kw_only=True)
class FallbackEvent(TelemetryEvent):
    event_type: ClassVar[TelemetryEventType] = TelemetryEventType.FALLBACK
    ladder_step: str
    trigger: str
    stabilization_mode: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class PressureSnapshotEvent(TelemetryEvent):
    event_type: ClassVar[TelemetryEventType] = TelemetryEventType.PRESSURE
    budget_bytes: int
    used_bytes: int
    pressure_level: str


@dataclass(frozen=True, slots=True, kw_only=True)
class SecurityEvent(TelemetryEvent):
    event_type: ClassVar[TelemetryEventType] = TelemetryEventType.SECURITY
    caller_id: str
    endpoint: str
    decision: str


@dataclass(frozen=True, slots=True, kw_only=True)
class PolicyEvent(TelemetryEvent):
    event_type: ClassVar[TelemetryEventType] = TelemetryEventType.POLICY
    previous_profile: str
    new_profile: str


@dataclass(frozen=True, slots=True)
class TelemetryRecord:
    """Persisted redacted telemetry record."""

    record_class: TelemetryRecordClass
    event_type: TelemetryEventType
    reason_code: TelemetryReasonCode
    correlation_id: str
    timestamp: datetime
    session_id_hash: str | None
    identifiers: dict[str, str]
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_class": self.record_class.value,
            "event_type": self.event_type.value,
            "reason_code": self.reason_code.value,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp.isoformat(),
            "session_id_hash": self.session_id_hash,
            "identifiers": dict(self.identifiers),
            "payload": _json_safe(self.payload),
        }


@dataclass(frozen=True, slots=True)
class TelemetryExportBundle:
    """Local export bundle containing only redacted persisted records."""

    schema_version: str
    created_at: datetime
    policy: dict[str, Any]
    records: tuple[TelemetryRecord, ...]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at.isoformat(),
            "policy": _json_safe(self.policy),
            "summary": _json_safe(self.summary),
            "records": [record.to_dict() for record in self.records],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=True, indent=indent)


def _default_retention_seconds() -> dict[TelemetryRecordClass, int]:
    return {
        TelemetryRecordClass.EVENT: 24 * 60 * 60,
        TelemetryRecordClass.METRIC: 30 * 24 * 60 * 60,
        TelemetryRecordClass.CRASH: 14 * 24 * 60 * 60,
        TelemetryRecordClass.BUNDLE: 7 * 24 * 60 * 60,
    }


def _default_max_retention_seconds() -> dict[TelemetryRecordClass, int]:
    return {
        TelemetryRecordClass.EVENT: 72 * 60 * 60,
        TelemetryRecordClass.METRIC: 90 * 24 * 60 * 60,
        TelemetryRecordClass.CRASH: 30 * 24 * 60 * 60,
        TelemetryRecordClass.BUNDLE: 30 * 24 * 60 * 60,
    }


@dataclass(slots=True)
class TelemetryPolicy:
    """Policy container for local-only telemetry behavior."""

    local_only: bool = True
    export_opt_in: bool = False
    allow_debug_identifiers: bool = False
    identifier_hash_salt: str = ""
    retention_seconds_by_class: dict[TelemetryRecordClass, int] = field(
        default_factory=_default_retention_seconds
    )
    max_retention_seconds_by_class: dict[TelemetryRecordClass, int] = field(
        default_factory=_default_max_retention_seconds
    )

    def validate(self) -> None:
        for record_class, retention in self.retention_seconds_by_class.items():
            max_retention = self.max_retention_seconds_by_class.get(record_class)
            if max_retention is None:
                continue
            if retention < 0:
                raise ApiError(
                    ApiErrorCode.INVALID_ARGUMENT,
                    f"Retention for {record_class.value} cannot be negative.",
                )
            if retention > max_retention:
                raise ApiError(
                    ApiErrorCode.INVALID_ARGUMENT,
                    f"Retention for {record_class.value} exceeds allowed maximum.",
                )

    def snapshot(self) -> dict[str, Any]:
        return {
            "local_only": self.local_only,
            "export_opt_in": self.export_opt_in,
            "allow_debug_identifiers": self.allow_debug_identifiers,
            "identifier_hash_salt": "<redacted>" if self.identifier_hash_salt else "",
            "retention_seconds_by_class": {
                key.value: value for key, value in self.retention_seconds_by_class.items()
            },
            "max_retention_seconds_by_class": {
                key.value: value for key, value in self.max_retention_seconds_by_class.items()
            },
        }


@dataclass(slots=True)
class TelemetryPipeline:
    """In-memory telemetry pipeline with redaction, retention, and export."""

    policy: TelemetryPolicy = field(default_factory=TelemetryPolicy)
    schema_version: str = "1.0"
    _records: list[TelemetryRecord] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self.policy.validate()

    @property
    def records(self) -> tuple[TelemetryRecord, ...]:
        return tuple(self._records)

    def set_export_opt_in(self, enabled: bool) -> None:
        self.policy.export_opt_in = bool(enabled)

    def record_event(
        self,
        event: TelemetryEvent,
        *,
        extra_identifiers: Mapping[str, Any] | None = None,
    ) -> TelemetryRecord:
        """Persist a structured event after redaction and identifier hashing."""

        structured = event.structured_dict()
        payload = {
            key: value
            for key, value in structured.items()
            if key
            not in {
                "event_type",
                "record_class",
                "reason_code",
                "correlation_id",
                "timestamp",
                "session_id",
            }
        }
        payload = redact_mapping(
            payload,
            salt=self.policy.identifier_hash_salt,
            allow_debug_identifiers=self.policy.allow_debug_identifiers,
        )

        identifiers = {
            key: (
                value
                if self.policy.allow_debug_identifiers
                else hash_identifier(value, salt=self.policy.identifier_hash_salt)
            )
            for key, value in (extra_identifiers or {}).items()
            if value is not None
        }
        session_id_hash = (
            hash_identifier(event.session_id, salt=self.policy.identifier_hash_salt)
            if event.session_id is not None and not self.policy.allow_debug_identifiers
            else event.session_id
        )
        if event.session_id is not None and self.policy.allow_debug_identifiers:
            session_id_hash = event.session_id

        record = TelemetryRecord(
            record_class=event.record_class,
            event_type=event.event_type,
            reason_code=event.reason_code,
            correlation_id=event.correlation_id,
            timestamp=event.timestamp,
            session_id_hash=session_id_hash,
            identifiers=identifiers,
            payload=payload,
        )
        self._records.append(record)
        return record

    def cleanup(
        self,
        *,
        now: datetime | None = None,
        on_remove: Callable[[TelemetryRecord], None] | None = None,
    ) -> tuple[TelemetryRecord, ...]:
        """Drop expired records and call the optional cleanup hook."""

        current_time = _coerce_datetime(now)
        kept: list[TelemetryRecord] = []
        removed: list[TelemetryRecord] = []

        for record in self._records:
            retention = self.policy.retention_seconds_by_class.get(record.record_class)
            if retention is None:
                kept.append(record)
                continue
            age_seconds = (current_time - record.timestamp).total_seconds()
            if age_seconds > retention:
                removed.append(record)
                if on_remove is not None:
                    on_remove(record)
            else:
                kept.append(record)

        self._records = kept
        return tuple(removed)

    def build_export_bundle(self, *, now: datetime | None = None) -> TelemetryExportBundle:
        """Build a local export bundle from already-redacted persisted records."""

        if not self.policy.export_opt_in:
            raise ApiError(
                ApiErrorCode.INVALID_STATE,
                "Telemetry export is disabled until explicit opt-in.",
            )

        current_time = _coerce_datetime(now)
        self.cleanup(now=current_time, on_remove=lambda _record: None)

        summary = {
            "record_count": len(self._records),
            "by_event_type": {},
            "by_reason_code": {},
        }
        for record in self._records:
            summary["by_event_type"][record.event_type.value] = (
                summary["by_event_type"].get(record.event_type.value, 0) + 1
            )
            summary["by_reason_code"][record.reason_code.value] = (
                summary["by_reason_code"].get(record.reason_code.value, 0) + 1
            )

        return TelemetryExportBundle(
            schema_version=self.schema_version,
            created_at=current_time,
            policy=self.policy.snapshot(),
            records=tuple(self._records),
            summary=summary,
        )


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


__all__ = [
    "FallbackEvent",
    "PolicyEvent",
    "PressureSnapshotEvent",
    "ResidencyChangeEvent",
    "SecurityEvent",
    "TelemetryEvent",
    "TelemetryEventType",
    "TelemetryExportBundle",
    "TelemetryPipeline",
    "TelemetryPolicy",
    "TelemetryReasonCode",
    "TelemetryRecord",
    "TelemetryRecordClass",
    "TransferEvent",
    "hash_identifier",
    "new_correlation_id",
    "redact_mapping",
    "redact_text",
    "redact_value",
]
