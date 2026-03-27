"""Stable error codes and exception types for AstraWeave."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ApiErrorCode(str, Enum):
    """Stable API error codes for the service and SDK."""

    OK = "AW_OK"
    AUTH_DENIED = "AW_ERR_AUTH_DENIED"
    RATE_LIMITED = "AW_ERR_RATE_LIMITED"
    INVALID_ARGUMENT = "AW_ERR_INVALID_ARGUMENT"
    INVALID_STATE = "AW_ERR_INVALID_STATE"
    NOT_FOUND = "AW_ERR_NOT_FOUND"
    TIMEOUT = "AW_ERR_TIMEOUT"
    CANCELLED = "AW_ERR_CANCELLED"
    RESOURCE_EXHAUSTED = "AW_ERR_RESOURCE_EXHAUSTED"
    CONFLICT_RUN_IN_PROGRESS = "AW_ERR_CONFLICT_RUN_IN_PROGRESS"
    DRIVER_BUDGET_CHANGED = "AW_ERR_DRIVER_BUDGET_CHANGED"
    UNSUPPORTED_CAPABILITY = "AW_ERR_UNSUPPORTED_CAPABILITY"
    INTERNAL = "AW_ERR_INTERNAL"


@dataclass(frozen=True, slots=True)
class ApiError(Exception):
    """Typed exception that carries a stable AstraWeave API error code."""

    code: ApiErrorCode
    message: str

    def __post_init__(self) -> None:
        # M11 fix: populate Exception.args so repr() and standard handlers work
        object.__setattr__(self, "args", (self.code, self.message))

    def __str__(self) -> str:
        return f"{self.code.value}: {self.message}"

