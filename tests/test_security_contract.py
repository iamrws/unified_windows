"""Contract tests for caller authorization and rate limiting."""

from __future__ import annotations

import importlib
from inspect import signature
import os
import unittest


class _FakeClock:
    """Deterministic clock for rate-limit and session-cap tests."""

    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


class SecurityContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.security = importlib.import_module("astrawave.security")
        except ModuleNotFoundError as exc:  # pragma: no cover - makes the failure obvious
            raise AssertionError("astrawave.security is required for the security contract tests") from exc

        try:
            cls.CallerIdentity = getattr(cls.security, "CallerIdentity")
            cls.SecurityPolicy = getattr(cls.security, "SecurityPolicy")
            cls.SecurityGuard = getattr(cls.security, "SecurityGuard")
            cls.SecurityDecision = getattr(cls.security, "SecurityDecision")
            cls.SecurityDenyReason = getattr(cls.security, "SecurityDenyReason")
            cls.attest_caller_identity = getattr(cls.security, "attest_caller_identity")
            cls.resolve_process_user_sid = getattr(cls.security, "resolve_process_user_sid")
        except AttributeError as exc:  # pragma: no cover - keeps the contract explicit
            raise AssertionError(
                "astrawave.security must expose CallerIdentity, SecurityPolicy, SecurityGuard, "
                "SecurityDecision, SecurityDenyReason, attest_caller_identity, and resolve_process_user_sid"
            ) from exc

    def _make_policy(self, **policy_overrides):
        kwargs = dict(policy_overrides)
        params = signature(self.SecurityPolicy).parameters
        if "service_owner_sid" in params and "service_owner_sid" not in kwargs:
            kwargs["service_owner_sid"] = "S-1-5-21-1000"
        filtered = {name: value for name, value in kwargs.items() if name in params}
        return self.SecurityPolicy(**filtered)

    def _make_guard(self, **policy_overrides):
        policy = self._make_policy(**policy_overrides)
        return self.SecurityGuard(policy)

    def _call(self, obj, method_names, *args, **kwargs):
        for name in method_names:
            if hasattr(obj, name):
                method = getattr(obj, name)
                params = signature(method).parameters
                if "caller" in params and args:
                    return method(*args, **kwargs)
                filtered_kwargs = {key: value for key, value in kwargs.items() if key in params}
                if len(args) == 1 and any(param.kind.name == "KEYWORD_ONLY" for param in params.values()):
                    return method(*args, **filtered_kwargs)
                return method(*args, **filtered_kwargs)
        self.fail(f"{obj.__class__.__name__} is missing one of the required methods: {method_names}")

    def test_same_user_authorization_is_allowed(self) -> None:
        gate = self._make_guard()
        caller = self.CallerIdentity(user_sid="S-1-5-21-1000", pid=101)

        decision = self._call(gate, ("authorize_caller",), caller)
        self.assertTrue(decision.allowed)
        self.assertIsNone(decision.reason)
        self.assertEqual(decision.error_code.value, "AW_OK")

    def test_unknown_or_cross_user_caller_is_denied_by_default(self) -> None:
        gate = self._make_guard()
        caller = self.CallerIdentity(user_sid="S-1-5-21-2000", pid=202)

        decision = self._call(gate, ("authorize_caller",), caller)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason.name, "UNKNOWN_CALLER")
        self.assertEqual(decision.error_code.value, "AW_ERR_AUTH_DENIED")

    def test_cross_user_allowlist_can_grant_access(self) -> None:
        gate = self._make_guard(allowed_cross_user_sids=frozenset({"S-1-5-21-2000"}))
        cross_user_caller = self.CallerIdentity(user_sid="S-1-5-21-2000", pid=304)

        decision = self._call(gate, ("authorize_caller",), cross_user_caller)
        self.assertTrue(decision.allowed)

    def test_create_session_rate_limit_and_session_cap_are_enforced(self) -> None:
        clock = _FakeClock()
        gate = self.SecurityGuard(
            self._make_policy(create_session_limit_per_minute=100),
            clock=clock,
        )
        caller = self.CallerIdentity(user_sid="S-1-5-21-1000", pid=404)

        for _ in range(8):
            decision = gate.admit_create_session(caller)
            self.assertTrue(decision.allowed)

        denied = gate.admit_create_session(caller)
        self.assertFalse(denied.allowed)
        self.assertEqual(denied.reason.name, "MAX_CONCURRENT_SESSIONS")
        self.assertEqual(denied.error_code.value, "AW_ERR_RESOURCE_EXHAUSTED")

    def test_create_session_rate_limit_is_below_unbounded_spam(self) -> None:
        gate = self.SecurityGuard(self._make_policy(), clock=_FakeClock())
        caller = self.CallerIdentity(user_sid="S-1-5-21-1000", pid=505)
        for _ in range(6):
            self.assertTrue(gate.admit_create_session(caller).allowed)

        denied = gate.admit_create_session(caller)
        self.assertFalse(denied.allowed)
        self.assertEqual(denied.reason.name, "RATE_LIMIT_EXCEEDED")

    def test_caller_identity_rejects_bool_pid(self) -> None:
        with self.assertRaises(ValueError):
            self.CallerIdentity(user_sid="S-1-5-21-1000", pid=True)

    def test_runtime_attestation_rejects_sid_mismatch(self) -> None:
        caller = self.CallerIdentity(user_sid="S-1-5-21-wrong", pid=os.getpid())
        decision = type(self).attest_caller_identity(
            caller,
            pid_lookup=lambda pid: True,
            sid_lookup=lambda pid: "S-1-5-21-right",
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason.name, "UNKNOWN_CALLER")
        self.assertEqual(decision.error_code.value, "AW_ERR_AUTH_DENIED")

    def test_runtime_attestation_rejects_inactive_pid(self) -> None:
        caller = self.CallerIdentity(user_sid="S-1-5-21-1000", pid=999999)
        decision = type(self).attest_caller_identity(
            caller,
            pid_lookup=lambda pid: False,
            sid_lookup=lambda pid: None,
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason.name, "UNKNOWN_CALLER")

    def test_runtime_attestation_accepts_live_pid_with_matching_sid(self) -> None:
        current_sid = type(self).resolve_process_user_sid(os.getpid()) or "S-1-5-21-1000"
        caller = self.CallerIdentity(user_sid=current_sid, pid=os.getpid())
        decision = type(self).attest_caller_identity(
            caller,
            pid_lookup=lambda pid: True,
            sid_lookup=lambda pid: current_sid,
        )
        self.assertTrue(decision.allowed)


if __name__ == "__main__":
    unittest.main()
