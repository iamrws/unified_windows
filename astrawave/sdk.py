"""High-level AstraWeave SDK wrapper.

The SDK is intentionally thin: it wraps an IPC client when available and
provides convenient defaults for caller identity, context-manager support, and
service-style method names.
"""

from __future__ import annotations

import inspect
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
        runtime_profile: str | None = None,
        runtime_backend_options: dict[str, Any] | None = None,
        caller_identity: CallerIdentity | None = None,
    ) -> dict[str, Any]:
        return self._invoke(
            "LoadModel",
            session_id,
            model_name,
            runtime_backend=runtime_backend,
            runtime_profile=runtime_profile,
            runtime_backend_options=runtime_backend_options,
            caller_identity=caller_identity,
        )

    def RegisterTensor(
        self,
        session_id: str,
        tensor_name: str,
        size_bytes: int,
        *,
        caller_identity: CallerIdentity | None = None,
    ) -> dict[str, Any]:
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
    ) -> dict[str, Any]:
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
    ) -> dict[str, Any]:
        return self._invoke("PrefetchPlan", session_id, caller_identity=caller_identity)

    def RunStep(
        self,
        session_id: str,
        step_name: str = "run",
        *,
        prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        runtime_profile_override: str | None = None,
        runtime_backend_options_override: dict[str, Any] | None = None,
        caller_identity: CallerIdentity | None = None,
    ) -> dict[str, Any]:
        return self._invoke(
            "RunStep",
            session_id,
            step_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            runtime_profile_override=runtime_profile_override,
            runtime_backend_options_override=runtime_backend_options_override,
            caller_identity=caller_identity,
        )

    def GetResidency(
        self,
        session_id: str,
        *,
        caller_identity: CallerIdentity | None = None,
    ) -> dict[str, Any]:
        return self._invoke("GetResidency", session_id, caller_identity=caller_identity)

    def GetPressure(
        self,
        session_id: str,
        *,
        caller_identity: CallerIdentity | None = None,
    ) -> dict[str, Any]:
        return self._invoke("GetPressure", session_id, caller_identity=caller_identity)

    def SetPolicy(
        self,
        session_id: str,
        policy: PolicyProfile,
        *,
        caller_identity: CallerIdentity | None = None,
    ) -> dict[str, Any]:
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
    ) -> None:
        self._invoke("CloseSession", session_id, caller_identity=caller_identity)

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
            caller_kwarg = self._detect_caller_param(method)
            if caller_kwarg is not None:
                kwargs[caller_kwarg] = caller
            return method(*args, **kwargs)

        call_fn = getattr(self._client, "call", None)
        if callable(call_fn):
            params = self._build_params(method_name, args, kwargs)
            return self._call_transport(call_fn, method_name, params, caller)

        raise RuntimeError(
            f"Wrapped client does not expose method '{method_name}' or a generic call interface."
        )

    @staticmethod
    def _detect_caller_param(method: Callable[..., Any]) -> str | None:
        """Inspect *method* to find which caller parameter it accepts.

        Returns ``'caller_identity'``, ``'caller'``, or ``None`` when the
        method accepts neither.  The result is determined via signature
        inspection so no trial-and-error calls are needed.
        """
        try:
            sig = inspect.signature(method)
        except (ValueError, TypeError):
            # Fallback for built-in / C-extension callables that have no
            # inspectable signature: prefer caller_identity by convention.
            return "caller_identity"
        params = sig.parameters
        if "caller_identity" in params:
            return "caller_identity"
        if "caller" in params:
            return "caller"
        # Check for **kwargs – the method may accept arbitrary keywords.
        for p in params.values():
            if p.kind == inspect.Parameter.VAR_KEYWORD:
                return "caller_identity"
        return None

    def _call_transport(
        self,
        call_fn: Callable[..., Any],
        method_name: str,
        params: dict[str, Any],
        caller: CallerIdentity | None,
    ) -> Any:
        try:
            sig = inspect.signature(call_fn)
        except (ValueError, TypeError):
            # Uninspectable (C-extension, etc.) -- fall back to the most common
            # positional convention: call(method_name, params, caller).
            return call_fn(method_name, params, caller)

        sig_params = sig.parameters
        param_names = list(sig_params.keys())
        has_var_positional = any(
            p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig_params.values()
        )
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_params.values()
        )

        # Determine positional args beyond method_name based on signature.
        positional_args: list[Any] = [method_name]
        # The second positional param (after method_name) is typically params.
        if has_var_positional or len(param_names) >= 2:
            positional_args.append(params)
        # Third positional param may be the caller.
        if has_var_positional or len(param_names) >= 3:
            positional_args.append(caller)

        # Build keyword args for caller / params when accepted as keywords.
        kw: dict[str, Any] = {}
        if "params" in sig_params and "params" not in kw:
            kw["params"] = params
        caller_key = (
            "caller_identity" if "caller_identity" in sig_params
            else "caller" if "caller" in sig_params
            else None
        )
        if caller_key is not None:
            kw[caller_key] = caller

        # Prefer keyword style when the signature explicitly names the params.
        if kw:
            return call_fn(method_name, **kw)

        # Otherwise fall back to positional.
        if has_var_positional or has_var_keyword:
            return call_fn(*positional_args)

        # Minimal call: just method_name.
        return call_fn(method_name)

    # -- Dispatch table for _build_params (M31) ---------------------------------
    # Each entry maps a method name to:
    #   positional_keys  - names assigned to positional args (in order)
    #   optional_kwargs  - keyword-arg names that are forwarded when not None

    _PARAM_SCHEMA: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
        "CreateSession": ((), ()),
        "LoadModel": (
            ("session_id", "model_name"),
            ("runtime_backend", "runtime_profile", "runtime_backend_options"),
        ),
        "RegisterTensor": (("session_id", "tensor_name", "size_bytes"), ()),
        "SetTierHint": (("session_id", "tensor_name", "tier"), ()),
        "PrefetchPlan": (("session_id",), ()),
        "RunStep": (
            ("session_id", "step_name"),
            ("prompt", "max_tokens", "temperature", "runtime_profile_override", "runtime_backend_options_override"),
        ),
        "GetResidency": (("session_id",), ()),
        "GetPressure": (("session_id",), ()),
        "SetPolicy": (("session_id", "policy"), ()),
        "CloseSession": (("session_id",), ()),
    }

    def _build_params(self, method_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
        schema = self._PARAM_SCHEMA.get(method_name)
        if schema is None:
            raise RuntimeError(f"Unsupported SDK method '{method_name}'.")
        positional_keys, optional_kwargs = schema
        params: dict[str, Any] = dict(zip(positional_keys, args))
        for key in optional_kwargs:
            value = kwargs.get(key)
            if value is not None:
                params[key] = value
        return params

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
