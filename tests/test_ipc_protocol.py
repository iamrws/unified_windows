"""Contract tests for the AstraWeave IPC protocol envelope."""

from __future__ import annotations

import importlib
import unittest


class IpcProtocolContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.protocol = importlib.import_module("astrawave.ipc_protocol")
        except ModuleNotFoundError as exc:  # pragma: no cover - makes the contract explicit
            raise unittest.SkipTest("astrawave.ipc_protocol is not present yet") from exc

        try:
            errors = importlib.import_module("astrawave.errors")
            cls.ApiError = errors.ApiError
            cls.ApiErrorCode = errors.ApiErrorCode
        except Exception as exc:  # pragma: no cover - defensive
            raise AssertionError("astrawave.errors must expose ApiError and ApiErrorCode") from exc

    def _lookup(self, *names: str):
        for name in names:
            if hasattr(self.protocol, name):
                return getattr(self.protocol, name)
        self.fail(f"astrawave.ipc_protocol is missing one of: {', '.join(names)}")

    def test_request_envelope_validation_accepts_well_formed_input(self) -> None:
        RequestEnvelope = self._lookup("RequestEnvelope")
        CallerEnvelope = self._lookup("CallerEnvelope")
        validate_request_payload = self._lookup("validate_request_payload")

        caller = CallerEnvelope(user_sid="S-1-5-21-1000", pid=1001)
        envelope = RequestEnvelope(
            id="req-1",
            method="CreateSession",
            params={},
            caller=caller,
        )
        validated = validate_request_payload(envelope.to_dict())

        self.assertEqual(validated.id, "req-1")
        self.assertEqual(validated.method, "CreateSession")
        self.assertEqual(validated.caller.user_sid, caller.user_sid)
        self.assertEqual(validated.caller.pid, caller.pid)

    def test_request_envelope_validation_rejects_malformed_input(self) -> None:
        validate_request_payload = self._lookup("validate_request_payload")

        malformed_cases = [
            {},
            {"id": "", "method": "CreateSession", "params": {}},
            {"id": "req-1", "method": "", "params": {}},
            {"id": "req-1", "method": "CreateSession"},
            {"id": "req-1", "method": "CreateSession", "params": {"bad": object()}, "caller": None},
            {"id": "req-1", "method": "CreateSession", "params": {}, "caller": {"pid": 1}},
            {"id": "req-1", "method": "CreateSession", "params": {}, "caller": {"user_sid": "", "pid": 1}},
        ]

        for payload in malformed_cases:
            with self.subTest(payload=payload):
                with self.assertRaises(self.ApiError):
                    validate_request_payload(payload)

    def test_error_envelope_maps_api_error_codes_stably(self) -> None:
        ErrorResponse = self._lookup("ErrorResponse")
        ErrorPayload = self._lookup("ErrorPayload")
        error_response = self._lookup("error_response")
        api_error = self.ApiError(self.ApiErrorCode.AUTH_DENIED, "not allowed")

        envelope = error_response(api_error, id="req-7")
        payload = envelope.to_dict()

        self.assertEqual(payload["id"], "req-7")
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], self.ApiErrorCode.AUTH_DENIED.value)
        self.assertEqual(payload["error"]["message"], "not allowed")
        self.assertIsInstance(envelope, ErrorResponse)
        self.assertIsInstance(envelope.error, ErrorPayload)

    def test_success_envelope_is_json_serializable(self) -> None:
        SuccessResponse = self._lookup("SuccessResponse")
        response_to_json = self._lookup("response_to_json")

        envelope = SuccessResponse(id="req-9", result={"status": "ok"})
        rendered = response_to_json(envelope)

        self.assertIsInstance(rendered, str)
        self.assertIn('"ok": true', rendered.lower())
        self.assertIn('"id": "req-9"', rendered)

    def test_request_construction_includes_optional_caller(self) -> None:
        RequestEnvelope = self._lookup("RequestEnvelope")
        CallerEnvelope = self._lookup("CallerEnvelope")

        caller = CallerEnvelope(user_sid="S-1-5-21-2000", pid=2020)
        envelope = RequestEnvelope(
            id="req-11",
            method="LoadModel",
            params={"session_id": "s-1", "model_name": "m"},
            caller=caller,
        )
        payload = envelope.to_dict()

        self.assertEqual(payload["caller"]["user_sid"], caller.user_sid)
        self.assertEqual(payload["caller"]["pid"], caller.pid)

    def test_request_envelope_rejects_payloads_over_max_size(self) -> None:
        validate_request_payload = self._lookup("validate_request_payload")
        max_bytes = self._lookup("MAX_CONTROL_PLANE_PAYLOAD_BYTES")
        oversized_payload = {
            "id": "req-large",
            "method": "CreateSession",
            "params": {"blob": "x" * (int(max_bytes) + 1024)},
            "caller": {"user_sid": "S-1-5-21-1000", "pid": 1001},
        }

        with self.assertRaises(self.ApiError) as ctx:
            validate_request_payload(oversized_payload)
        self.assertEqual(ctx.exception.code, self.ApiErrorCode.INVALID_ARGUMENT)


if __name__ == "__main__":
    unittest.main()
