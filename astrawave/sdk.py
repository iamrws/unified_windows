"""High-level AstraWeave SDK wrapper.

The SDK is intentionally thin: it wraps an IPC client when available and
provides convenient defaults for caller identity, context-manager support, and
service-style method names.
"""

from __future__ import annotations

from typing import Any, Callable

from .security import CallerIdentity
from .types import MemoryTier, PolicyProfile

try:  # pragma: no cover - optional transport layer may land separately
    from .ipc_client import AstraWeaveIpcClient
except ImportError:  # pragma: no cover - graceful fallback when transport is absent
    AstraWeaveIpcClient = None


class AstraWeaveSDK:
    """Convenience wrapper around `AstraWeaveIpcClient`.

    Parameters
    ----------
    client:
        Optional pre-built IPC client instance. When omitted, the SDK will try
        to construct `AstraWeaveIpcClient` if that transport module is present.
    default_caller_identity:
        Default caller identity used when an API method is invoked without an
        explicit caller.
    default_user_sid / default_pid:
        Convenience fields for constructing a default caller identity when a
        full identity object is not supplied.
    client_factory:
        Optional factory for dependency injection. This is useful for tests or
        alternative transports.
    client_kwargs:
        Keyword arguments forwarded to the transport client factory/constructor.
    """

    def __init__(
        self,
        client: Any | None = None,
        *,
        default_caller_identity: CallerIdentity | None = None,
        default_user_sid: str | None = None,
        default_pid: int | None = None,
        client_factory: Callable[..., Any] | None = None,
        **client_kwargs: Any,
    ) -> None:
        self._client = client if client is not None else self._build_client(client_factory, client_kwargs)
        self._default_caller_identity = self._resolve_default_caller_identity(
            default_caller_identity,
            default_user_sid,
            default_pid,
        )
        self._closed = False

    @property
    def client(self) -> Any:
        """Expose the wrapped IPC client."""

        return self._client

    @property
    def default_caller_identity(self) -> CallerIdentity | None:
        """Return the configured default caller identity, if any."""

        return self._default_caller_identity

    def __enter__(self) -> "AstraWeaveSDK":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def close(self) -> None:
        """Close the wrapped transport client if it exposes a close method."""

        if self._closed:
            return
        self._closed = True
        for attr_name in ("close", "disconnect", "stop"):
            close_fn = getattr(self._client, attr_name, None)
            if callable(close_fn):
                close_fn()
                return

    def CreateSession(self, caller_identity: CallerIdentity | None = None) -> str:
        return self._invoke("CreateSession", caller_identity=caller_identity)

    def LoadModel(
        self,
        session_id: str,
        model_name: str,
        *,
        runtime_backend: str | None = None,
        caller_identity: CallerIdentity | None = None,
    ) -> Any:
        return self._invoke(
            "LoadModel",
            session_id,
            model_name,
            runtime_backend=runtime_backend,
            caller_identity=caller_identity,
        )

    def RegisterTensor(
        self,
        session_id: str,
        tensor_name: str,
        size_bytes: int,
        *,
        caller_identity: CallerIdentity | None = None,
    ) -> Any:
        return self._invoke(
            "RegisterTensor",
            session_id,
            tensor_name,
            size_bytes,
            caller_identity=caller_identity,
        )

    def SetTierHint(
        self,
        session_id: str,
        tensor_name: str,
        tier: MemoryTier,
        *,
        caller_identity: CallerIdentity | None = None,
    ) -> Any:
        return self._invoke(
            "SetTierHint",
            session_id,
            tensor_name,
            tier,
            caller_identity=caller_identity,
        )

    def PrefetchPlan(
        self,
        session_id: str,
        *,
        caller_identity: CallerIdentity | None = None,
    ) -> Any:
        return self._invoke("PrefetchPlan", session_id, caller_identity=caller_identity)

    def RunStep(
        self,
        session_id: str,
        step_name: str = "run",
        *,
        prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        caller_identity: CallerIdentity | None = None,
    ) -> Any:
        return self._invoke(
            "RunStep",
            session_id,
            step_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            caller_identity=caller_identity,
        )

    def GetResidency(
        self,
        session_id: str,
        *,
        caller_identity: CallerIdentity | None = None,
    ) -> Any:
        return self._invoke("GetResidency", session_id, caller_identity=caller_identity)

    def GetPressure(
        self,
        session_id: str,
        *,
        caller_identity: CallerIdentity | None = None,
    ) -> Any:
        return self._invoke("GetPressure", session_id, caller_identity=caller_identity)

    def SetPolicy(
        self,
        session_id: str,
        policy: PolicyProfile,
        *,
        caller_identity: CallerIdentity | None = None,
    ) -> Any:
        return self._invoke(
            "SetPolicy",
            session_id,
            policy,
            caller_identity=caller_identity,
        )

    def CloseSession(
        self,
        session_id: str,
        *,
        caller_identity: CallerIdentity | None = None,
    ) -> Any:
        return self._invoke("CloseSession", session_id, caller_identity=caller_identity)

    def _invoke(
        self,
        method_name: str,
        *args: Any,
        caller_identity: CallerIdentity | None = None,
        **kwargs: Any,
    ) -> Any:
        caller = self._resolve_caller_identity(caller_identity)
        method = getattr(self._client, method_name, None)
        if callable(method):
            try:
                return method(*args, **kwargs, caller_identity=caller)
            except TypeError:
                try:
                    return method(*args, **kwargs, caller=caller)
                except TypeError:
                    return method(*args, **kwargs)

        call_fn = getattr(self._client, "call", None)
        if callable(call_fn):
            params = self._build_params(method_name, args, kwargs)
            return self._call_transport(call_fn, method_name, params, caller)

        raise RuntimeError(
            f"Wrapped client does not expose method '{method_name}' or a generic call interface."
        )

    def _call_transport(
        self,
        call_fn: Callable[..., Any],
        method_name: str,
        params: dict[str, Any],
        caller: CallerIdentity | None,
    ) -> Any:
        attempts = (
            (method_name, params, caller),
            (method_name, params),
            (method_name,),
        )
        keyword_attempts = (
            {"caller": caller, "params": params},
            {"caller_identity": caller, "params": params},
            {"caller": caller},
            {"caller_identity": caller},
            {"params": params},
            {},
        )

        for call_args in attempts:
            try:
                return call_fn(*call_args)
            except TypeError:
                pass
        for call_kwargs in keyword_attempts:
            try:
                return call_fn(method_name, **call_kwargs)
            except TypeError:
                pass
        raise RuntimeError(f"Transport client cannot dispatch method '{method_name}'.")

    def _build_params(self, method_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
        if method_name == "CreateSession":
            return {}
        if method_name == "LoadModel":
            session_id, model_name = args
            params = {"session_id": session_id, "model_name": model_name}
            runtime_backend = kwargs.get("runtime_backend")
            if runtime_backend is not None:
                params["runtime_backend"] = runtime_backend
            return params
        if method_name == "RegisterTensor":
            session_id, tensor_name, size_bytes = args
            return {
                "session_id": session_id,
                "tensor_name": tensor_name,
                "size_bytes": size_bytes,
            }
        if method_name == "SetTierHint":
            session_id, tensor_name, tier = args
            return {
                "session_id": session_id,
                "tensor_name": tensor_name,
                "tier": tier,
            }
        if method_name == "PrefetchPlan":
            (session_id,) = args
            return {"session_id": session_id}
        if method_name == "RunStep":
            session_id, step_name = args
            params = {"session_id": session_id, "step_name": step_name}
            prompt = kwargs.get("prompt")
            max_tokens = kwargs.get("max_tokens")
            temperature = kwargs.get("temperature")
            if prompt is not None:
                params["prompt"] = prompt
            if max_tokens is not None:
                params["max_tokens"] = max_tokens
            if temperature is not None:
                params["temperature"] = temperature
            return params
        if method_name == "GetResidency":
            (session_id,) = args
            return {"session_id": session_id}
        if method_name == "GetPressure":
            (session_id,) = args
            return {"session_id": session_id}
        if method_name == "SetPolicy":
            session_id, policy = args
            return {"session_id": session_id, "policy": policy}
        if method_name == "CloseSession":
            (session_id,) = args
            return {"session_id": session_id}
        raise RuntimeError(f"Unsupported SDK method '{method_name}'.")

    def _resolve_caller_identity(
        self,
        caller_identity: CallerIdentity | None,
    ) -> CallerIdentity | None:
        if caller_identity is not None:
            return caller_identity
        return self._default_caller_identity

    def _resolve_default_caller_identity(
        self,
        default_caller_identity: CallerIdentity | None,
        default_user_sid: str | None,
        default_pid: int | None,
    ) -> CallerIdentity | None:
        if default_caller_identity is not None:
            return default_caller_identity
        if default_user_sid is None and default_pid is None:
            return None
        if default_user_sid is None or default_pid is None:
            raise ValueError(
                "default_user_sid and default_pid must both be provided when constructing a default caller identity"
            )
        return CallerIdentity(default_user_sid, default_pid)

    def _build_client(
        self,
        client_factory: Callable[..., Any] | None,
        client_kwargs: dict[str, Any],
    ) -> Any:
        if client_factory is not None:
            return client_factory(**client_kwargs)
        if AstraWeaveIpcClient is None:
            raise RuntimeError(
                "AstraWeaveIpcClient is not available; provide a client instance or client_factory."
            )
        return AstraWeaveIpcClient(**client_kwargs)


__all__ = ["AstraWeaveSDK"]
