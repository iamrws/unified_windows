"""Contract tests for the AstraWeave IPC server adapter."""

from __future__ import annotations

import importlib
import unittest


class IpcServerContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.server_module = importlib.import_module("astrawave.ipc_server")
            cls.protocol_module = importlib.import_module("astrawave.ipc_protocol")
            cls.service_module = importlib.import_module("astrawave.service")
            cls.security_module = importlib.import_module("astrawave.security")
        except ModuleNotFoundError as exc:  # pragma: no cover - explicit skip path
            raise unittest.SkipTest("astrawave.ipc_server / ipc_protocol are not present yet") from exc

        errors = importlib.import_module("astrawave.errors")
        cls.ApiError = errors.ApiError
        cls.ApiErrorCode = errors.ApiErrorCode

    def _lookup(self, module, *names: str):
        for name in names:
            if hasattr(module, name):
                return getattr(module, name)
        self.fail(f"{module.__name__} is missing one of: {', '.join(names)}")

    def _make_server(self):
        server_cls = self._lookup(self.server_module, "AstraWeaveIpcServer")
        service_cls = getattr(self.service_module, "AstraWeaveService")
        return server_cls(service_cls())

    def _make_caller(self, user_sid: str = "S-1-5-21-1000", pid: int = 1001):
        caller_cls = self._lookup(self.security_module, "CallerIdentity")
        return caller_cls(user_sid=user_sid, pid=pid)

    def _make_request(self, method: str, params: dict, caller=None, request_id: str = "req-1"):
        request_cls = self._lookup(self.server_module, "IpcRequestEnvelope")
        return request_cls(id=request_id, method=method, params=params, caller=caller)

    def _dispatch(self, server, envelope):
        for name in ("handle_request", "dispatch", "process_request", "serve_envelope"):
            if hasattr(server, name):
                return getattr(server, name)(envelope)
        self.fail("IPC server is missing a request dispatch entrypoint")

    def test_create_session_load_model_register_run_close_happy_path(self) -> None:
        server = self._make_server()
        caller = self._make_caller(user_sid=server.service.security_guard.policy.service_owner_sid, pid=1001)

        create_response = self._dispatch(server, self._make_request("CreateSession", {}, caller=caller, request_id="req-1"))
        self.assertTrue(create_response["ok"])
        session_id = create_response["result"]
        self.assertIsInstance(session_id, str)

        load_response = self._dispatch(
            server,
            self._make_request(
                "LoadModel",
                {"session_id": session_id, "model_name": "demo-model"},
                caller=caller,
                request_id="req-2",
            ),
        )
        self.assertTrue(load_response["ok"])

        tensor_response = self._dispatch(
            server,
            self._make_request(
                "RegisterTensor",
                {"session_id": session_id, "tensor_name": "kv", "size_bytes": 1024},
                caller=caller,
                request_id="req-3",
            ),
        )
        self.assertTrue(tensor_response["ok"])

        run_response = self._dispatch(
            server,
            self._make_request(
                "RunStep",
                {"session_id": session_id, "step_name": "decode"},
                caller=caller,
                request_id="req-4",
            ),
        )
        self.assertTrue(run_response["ok"])
        self.assertEqual(run_response["result"]["session_id"], session_id)
        self.assertEqual(run_response["result"]["step_name"], "decode")

        close_response = self._dispatch(
            server,
            self._make_request("CloseSession", {"session_id": session_id}, caller=caller, request_id="req-5"),
        )
        self.assertTrue(close_response["ok"])

    def test_caller_propagation_and_typed_denial_behavior(self) -> None:
        server = self._make_server()
        owner = self._make_caller(user_sid=server.service.security_guard.policy.service_owner_sid, pid=101)
        foreign = self._make_caller(user_sid="S-1-5-21-2000", pid=202)

        session_response = self._dispatch(
            server,
            self._make_request("CreateSession", {}, caller=owner, request_id="req-10"),
        )
        session_id = session_response["result"]

        denial = self._dispatch(
            server,
            self._make_request(
                "LoadModel",
                {"session_id": session_id, "model_name": "demo-model"},
                caller=foreign,
                request_id="req-11",
            ),
        )
        self.assertFalse(denial["ok"])
        self.assertEqual(denial["error"]["code"], self.ApiErrorCode.AUTH_DENIED.value)
        self.assertIsInstance(denial["error"]["message"], str)

    def test_server_returns_protocol_errors_for_malformed_payloads(self) -> None:
        server = self._make_server()
        malformed_envelopes = [
            {},
            {"id": "req-1"},
            {"id": "req-1", "method": "CreateSession", "params": "bad"},
            {"id": "req-1", "method": "CreateSession", "params": {}, "caller": {"pid": 1}},
            {"id": "", "method": "CreateSession", "params": {}},
        ]

        for envelope in malformed_envelopes:
            with self.subTest(envelope=envelope):
                response = self._dispatch(server, envelope)
                self.assertFalse(response["ok"])
                self.assertIn(response["error"]["code"], {code.value for code in self.ApiErrorCode})

    def test_server_passes_caller_identity_through_to_service(self) -> None:
        server = self._make_server()
        caller = self._make_caller(user_sid=server.service.security_guard.policy.service_owner_sid, pid=303)

        response = self._dispatch(
            server,
            self._make_request("CreateSession", {}, caller=caller, request_id="req-20"),
        )
        self.assertTrue(response["ok"])

        service = getattr(server, "service", None)
        self.assertIsNotNone(service)

        sessions = getattr(service, "_sessions", {})
        self.assertEqual(len(sessions), 1)
        stored_session = next(iter(sessions.values()))
        self.assertEqual(getattr(stored_session.owner_identity, "user_sid"), caller.user_sid)
        self.assertEqual(getattr(stored_session.owner_identity, "pid"), caller.pid)

    def test_server_requires_explicit_caller_identity_by_default(self) -> None:
        server = self._make_server()
        response = self._dispatch(
            server,
            self._make_request("CreateSession", {}, caller=None, request_id="req-30"),
        )
        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["code"], self.ApiErrorCode.AUTH_DENIED.value)

    def test_server_rejects_caller_switch_within_bound_connection_context(self) -> None:
        server = self._make_server()
        owner_sid = server.service.security_guard.policy.service_owner_sid
        bound = self._make_caller(user_sid=owner_sid, pid=401)
        switched = self._make_caller(user_sid=owner_sid, pid=402)

        response = server.handle_request(
            self._make_request("CreateSession", {}, caller=switched, request_id="req-31"),
            expected_caller=bound,
        )
        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["code"], self.ApiErrorCode.AUTH_DENIED.value)

    def test_server_rejects_non_localhost_listener_configuration(self) -> None:
        server_cls = self._lookup(self.server_module, "AstraWeaveIpcServer")
        service_cls = getattr(self.service_module, "AstraWeaveService")
        server = server_cls(service_cls(), prefer_named_pipe=False, host="0.0.0.0", port=0)
        with self.assertRaises(self.ApiError) as ctx:
            server.start()
        self.assertEqual(ctx.exception.code, self.ApiErrorCode.INVALID_ARGUMENT)

    def test_server_rejects_oversized_request_payload(self) -> None:
        server = self._make_server()
        max_bytes = self._lookup(self.protocol_module, "MAX_CONTROL_PLANE_PAYLOAD_BYTES")
        owner_sid = server.service.security_guard.policy.service_owner_sid
        payload = {
            "id": "req-too-large",
            "method": "CreateSession",
            "params": {"blob": "x" * (int(max_bytes) + 1024)},
            "caller": {"user_sid": owner_sid, "pid": 1001},
        }
        response = self._dispatch(server, payload)
        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["code"], self.ApiErrorCode.INVALID_ARGUMENT.value)

    def test_security_denials_are_recorded_in_telemetry(self) -> None:
        server = self._make_server()
        foreign = self._make_caller(user_sid="S-1-5-21-9999", pid=601)
        response = self._dispatch(
            server,
            self._make_request("CreateSession", {}, caller=foreign, request_id="req-security-deny"),
        )
        self.assertFalse(response["ok"])
        records = tuple(server.service.telemetry.records)
        self.assertTrue(
            any(
                record.event_type.value == "security_event"
                and record.reason_code.value == "AW_TEL_REASON_SECURITY_DENY"
                for record in records
            )
        )

    def test_server_accepts_canonical_protocol_request_envelope(self) -> None:
        server = self._make_server()
        RequestEnvelope = self._lookup(self.protocol_module, "RequestEnvelope")
        CallerEnvelope = self._lookup(self.protocol_module, "CallerEnvelope")
        owner_sid = server.service.security_guard.policy.service_owner_sid
        request = RequestEnvelope(
            id="req-32",
            method="CreateSession",
            params={},
            caller=CallerEnvelope(user_sid=owner_sid, pid=777),
        )

        response = self._dispatch(server, request)
        self.assertTrue(response["ok"])
        self.assertIsInstance(response["result"], str)


if __name__ == "__main__":
    unittest.main()
