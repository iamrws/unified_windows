"""AstraWeave package surface."""

from .errors import ApiError, ApiErrorCode
from .types import (
    MemoryTier,
    PolicyProfile,
    PressureSnapshot,
    ResidencySnapshot,
    ResidencyState,
    SessionState,
    TransferEvent,
    FallbackEvent,
)

try:
    from .service import AstraWeaveService
except ImportError:  # pragma: no cover - service is added by another worker
    AstraWeaveService = None

from .sdk import AstraWeaveSDK

try:  # pragma: no cover - optional IPC transport modules may not exist yet
    from .ipc_client import AstraWeaveIpcClient
except ImportError:  # pragma: no cover
    AstraWeaveIpcClient = None

try:  # pragma: no cover - optional IPC transport modules may not exist yet
    from .ipc_server import AstraWeaveIpcServer
except ImportError:  # pragma: no cover
    AstraWeaveIpcServer = None

__all__ = [
    "ApiError",
    "ApiErrorCode",
    "AstraWeaveSDK",
    "FallbackEvent",
    "MemoryTier",
    "PolicyProfile",
    "PressureSnapshot",
    "ResidencySnapshot",
    "ResidencyState",
    "SessionState",
    "TransferEvent",
]

if AstraWeaveService is not None:
    __all__.append("AstraWeaveService")
if AstraWeaveIpcClient is not None:
    __all__.append("AstraWeaveIpcClient")
if AstraWeaveIpcServer is not None:
    __all__.append("AstraWeaveIpcServer")
