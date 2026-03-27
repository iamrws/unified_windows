"""End-to-end IPC client and SDK contract tests for AstraWeave."""

from __future__ import annotations

import os
import time
import unittest

from astrawave.errors import ApiError, ApiErrorCode
from astrawave.ipc_client import AstraWeaveIpcClient
from astrawave.ipc_server import AstraWeaveIpcServer
from astrawave.sdk import AstraWeaveSDK
from astrawave.security import CallerIdentity, SecurityGuard, SecurityPolicy, resolve_process_user_sid
from astrawave.service import AstraWeaveService
from astrawave.types import MemoryTier, PolicyProfile


def _endpoint_to_uri(endpoint: object) -> str:
    if isinstance(endpoint, tuple) and len(endpoint) == 2:
        host, port = endpoint
        return f"tcp://{host}:{port}"
    if isinstance(endpoint, str):
        return endpoint if endpoint.startswith(("tcp://", "pipe://")) else f"tcp://{endpoint}"
    raise AssertionError(f"Unsupported endpoint shape: {endpoint!r}")


def _connect_client(endpoint: str, caller: CallerIdentity) -> AstraWeaveIpcClient:
    client = AstraWeaveIpcClient(
        endpoint=endpoint,
        timeout=5.0,
        default_caller=caller,
        prefer_named_pipe=False,
    )
    last_error: Exception | None = None
    for _ in range(40):
        try:
            client.connect()
            return client
        except (ConnectionError, OSError) as exc:  # H22 fix: only catch connection errors
            last_error = exc
            time.sleep(0.05)
    if last_error is not None:
        raise last_error
    raise AssertionError("client failed to connect")


class _SDKBridge:
    """Adapter that lets `AstraWeaveSDK` drive a real IPC client."""

    def __init__(self, client: AstraWeaveIpcClient) -> None:
        self._client = client

    def close(self) -> None:
        self._client.close()

    def CreateSession(self, caller_identity: CallerIdentity | None = None) -> str:
        return self._client.CreateSession(caller_identity)

    def LoadModel(
        self,
        session_id: str,
        model_name: str,
        caller_identity: CallerIdentity | None = None,
        runtime_backend: str | None = None,
        runtime_profile: str | None = None,
        runtime_backend_options: dict[str, object] | None = None,
    ) -> None:
        self._client.LoadModel(
            session_id,
            model_name,
            caller_identity,
            runtime_backend=runtime_backend,
            runtime_profile=runtime_profile,
            runtime_backend_options=runtime_backend_options,
        )

    def RegisterTensor(
        self,
        session_id: str,
        tensor_name: str,
        size_bytes: int,
        caller_identity: CallerIdentity | None = None,
    ) -> None:
        self._client.RegisterTensor(session_id, tensor_name, size_bytes, caller_identity)

    def SetTierHint(
        self,
        session_id: str,
        tensor_name: str,
        tier: MemoryTier,
        caller_identity: CallerIdentity | None = None,
    ) -> None:
        self._client.SetTierHint(session_id, tensor_name, tier, caller_identity)

    def PrefetchPlan(self, session_id: str, caller_identity: CallerIdentity | None = None):
        return self._client.PrefetchPlan(session_id, caller_identity)

    def RunStep(
        self,
        session_id: str,
        step_name: str = "run",
        caller_identity: CallerIdentity | None = None,
        prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        runtime_profile_override: str | None = None,
        runtime_backend_options_override: dict[str, object] | None = None,
    ):
        return self._client.RunStep(
            session_id,
            step_name,
            caller_identity,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            runtime_profile_override=runtime_profile_override,
            runtime_backend_options_override=runtime_backend_options_override,
        )

    def GetResidency(self, session_id: str, caller_identity: CallerIdentity | None = None):
        return self._client.GetResidency(session_id, caller_identity)

    def GetPressure(self, session_id: str, caller_identity: CallerIdentity | None = None):
        return self._client.GetPressure(session_id, caller_identity)

    def SetPolicy(
        self,
        session_id: str,
        policy: PolicyProfile,
        caller_identity: CallerIdentity | None = None,
    ) -> None:
        self._client.SetPolicy(session_id, policy, caller_identity)

    def CloseSession(self, session_id: str, caller_identity: CallerIdentity | None = None) -> None:
        self._client.CloseSession(session_id, caller_identity)


class IpcClientSdkE2ETests(unittest.TestCase):
    def setUp(self) -> None:
        current_pid = os.getpid()
        current_sid = resolve_process_user_sid(current_pid) or "S-1-5-21-1000"
        self.owner = CallerIdentity(user_sid=current_sid, pid=current_pid)
        self.foreign = CallerIdentity(user_sid="S-1-5-21-2000", pid=current_pid)

        service = AstraWeaveService(
            security_guard=SecurityGuard(
                SecurityPolicy(service_owner_sid=self.owner.user_sid)
            )
        )
        self.server = AstraWeaveIpcServer(
            service=service,
            prefer_named_pipe=False,
            host="127.0.0.1",
            port=0,
        )
        self.server.start()
        self.addCleanup(self.server.stop)

        self.endpoint = self._wait_for_endpoint()

    def _wait_for_endpoint(self) -> str:
        last_endpoint: object | None = None
        for _ in range(40):
            last_endpoint = self.server.endpoint
            if last_endpoint:
                return _endpoint_to_uri(last_endpoint)
            time.sleep(0.05)
        raise AssertionError(f"server did not expose an endpoint; last value={last_endpoint!r}")

    def _make_client(self, caller: CallerIdentity) -> AstraWeaveIpcClient:
        client = _connect_client(self.endpoint, caller)
        self.addCleanup(client.close)
        return client

    def _make_sdk(self, caller: CallerIdentity) -> AstraWeaveSDK:
        client = _connect_client(self.endpoint, caller)
        self.addCleanup(client.close)
        sdk = AstraWeaveSDK(
            client=_SDKBridge(client),
            default_caller_identity=caller,
        )
        self.addCleanup(sdk.close)
        return sdk

    def test_client_and_sdk_happy_path(self) -> None:
        direct_client = self._make_client(self.owner)
        session_id = direct_client.CreateSession(self.owner)
        direct_client.LoadModel(session_id, "demo-model", self.owner)
        direct_client.RegisterTensor(session_id, "kv", 1024, self.owner)
        direct_client.SetTierHint(session_id, "kv", MemoryTier.HOT, self.owner)

        prefetched = direct_client.PrefetchPlan(session_id, self.owner)
        self.assertIsInstance(prefetched, list)

        step = direct_client.RunStep(session_id, "decode", self.owner)
        self.assertEqual(step["session_id"], session_id)
        self.assertEqual(step["step_name"], "decode")

        residency = direct_client.GetResidency(session_id, self.owner)
        pressure = direct_client.GetPressure(session_id, self.owner)
        self.assertEqual(residency.session_id, session_id)
        self.assertEqual(pressure.session_id, session_id)

        direct_client.CloseSession(session_id, self.owner)
        direct_client.close()

        sdk = self._make_sdk(self.owner)
        sdk_session_id = sdk.CreateSession()
        sdk.LoadModel(sdk_session_id, "sdk-model")
        sdk.RegisterTensor(sdk_session_id, "weights", 2048)
        sdk.SetTierHint(sdk_session_id, "weights", MemoryTier.WARM)
        sdk_step = sdk.RunStep(sdk_session_id, "decode")
        self.assertEqual(sdk_step["session_id"], sdk_session_id)
        self.assertEqual(sdk.GetResidency(sdk_session_id).session_id, sdk_session_id)
        self.assertEqual(sdk.GetPressure(sdk_session_id).session_id, sdk_session_id)

        sdk.CloseSession(sdk_session_id)
        sdk.close()

    def test_optional_runtime_backend_and_generation_knobs_round_trip(self) -> None:
        direct_client = self._make_client(self.owner)
        session_id = direct_client.CreateSession(self.owner)
        direct_client.LoadModel(session_id, "demo-model", self.owner, runtime_backend="ollama")
        direct_client.RegisterTensor(session_id, "kv", 1024, self.owner)
        direct_client.SetTierHint(session_id, "kv", MemoryTier.HOT, self.owner)

        run_result = direct_client.RunStep(
            session_id,
            "decode",
            self.owner,
            prompt="hello there",
            max_tokens=16,
            temperature=0.2,
        )
        self.assertEqual(run_result["session_id"], session_id)
        self.assertEqual(run_result["step_name"], "decode")
        direct_client.close()

        sdk = self._make_sdk(self.owner)
        sdk_session_id = sdk.CreateSession()
        sdk.LoadModel(sdk_session_id, "sdk-model", runtime_backend="ollama")
        sdk.RegisterTensor(sdk_session_id, "weights", 2048)
        sdk.SetTierHint(sdk_session_id, "weights", MemoryTier.WARM)
        sdk_step = sdk.RunStep(
            sdk_session_id,
            "decode",
            prompt="generate a short response",
            max_tokens=12,
            temperature=0.4,
        )
        self.assertEqual(sdk_step["session_id"], sdk_session_id)
        sdk.CloseSession(sdk_session_id)
        sdk.close()

    def test_runtime_tuning_controls_round_trip_and_validate_inputs(self) -> None:
        class _Transport:
            def __init__(self) -> None:
                self.payloads: list[dict[str, object]] = []

            def request(self, payload, timeout):
                self.payloads.append(payload)
                return {"id": payload["id"], "ok": True, "result": None}

            def close(self) -> None:
                return None

        client = AstraWeaveIpcClient(endpoint="tcp://127.0.0.1:8765", timeout=5.0, default_caller=self.owner, prefer_named_pipe=False)
        transport = _Transport()
        client._transport = transport  # type: ignore[attr-defined]

        client.LoadModel(
            "session-1",
            "qwen2.5:14b",
            self.owner,
            runtime_backend="ollama",
            runtime_profile="vram_constrained",
            runtime_backend_options={"num_ctx": 4096, "num_batch": 1},
        )
        client.RunStep(
            "session-1",
            "decode",
            self.owner,
            prompt="hello",
            max_tokens=32,
            temperature=0.25,
            runtime_profile_override="memory_saver",
            runtime_backend_options_override={"num_predict": 64, "top_p": 0.9},
        )
        self.assertEqual(transport.payloads[0]["params"]["runtime_profile"], "vram_constrained")
        self.assertEqual(transport.payloads[0]["params"]["runtime_backend_options"], {"num_ctx": 4096, "num_batch": 1})
        self.assertEqual(transport.payloads[1]["params"]["runtime_profile_override"], "memory_saver")
        self.assertEqual(transport.payloads[1]["params"]["runtime_backend_options_override"], {"num_predict": 64, "top_p": 0.9})

        with self.assertRaises(ApiError) as cm:
            client.LoadModel(
                "session-2",
                "demo-model",
                self.owner,
                runtime_backend_options={"num_ctx": 0},
            )
        self.assertEqual(cm.exception.code, ApiErrorCode.INVALID_ARGUMENT)

        class _Bridge:
            def __init__(self) -> None:
                self.calls: list[tuple[str, dict[str, object]]] = []

            def LoadModel(
                self,
                session_id: str,
                model_name: str,
                caller_identity: CallerIdentity | None = None,
                runtime_backend: str | None = None,
                runtime_profile: str | None = None,
                runtime_backend_options: dict[str, object] | None = None,
            ) -> None:
                self.calls.append(
                    (
                        "LoadModel",
                        {
                            "session_id": session_id,
                            "model_name": model_name,
                            "caller_identity": caller_identity,
                            "runtime_backend": runtime_backend,
                            "runtime_profile": runtime_profile,
                            "runtime_backend_options": runtime_backend_options,
                        },
                    )
                )

            def RunStep(
                self,
                session_id: str,
                step_name: str = "run",
                caller_identity: CallerIdentity | None = None,
                prompt: str | None = None,
                max_tokens: int | None = None,
                temperature: float | None = None,
                runtime_profile_override: str | None = None,
                runtime_backend_options_override: dict[str, object] | None = None,
            ) -> None:
                self.calls.append(
                    (
                        "RunStep",
                        {
                            "session_id": session_id,
                            "step_name": step_name,
                            "caller_identity": caller_identity,
                            "prompt": prompt,
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                            "runtime_profile_override": runtime_profile_override,
                            "runtime_backend_options_override": runtime_backend_options_override,
                        },
                    )
                )

        bridge = _Bridge()
        sdk = AstraWeaveSDK(client=bridge, default_caller_identity=self.owner)
        sdk.LoadModel(
            "session-3",
            "demo-model",
            runtime_backend="ollama",
            runtime_profile="vram_constrained",
            runtime_backend_options={"num_ctx": 4096},
        )
        sdk.RunStep(
            "session-3",
            "decode",
            prompt="hello",
            runtime_profile_override="memory_saver",
            runtime_backend_options_override={"num_predict": 32},
        )
        self.assertEqual(bridge.calls[0][1]["runtime_profile"], "vram_constrained")
        self.assertEqual(bridge.calls[0][1]["runtime_backend_options"], {"num_ctx": 4096})
        self.assertEqual(bridge.calls[1][1]["runtime_profile_override"], "memory_saver")
        self.assertEqual(bridge.calls[1][1]["runtime_backend_options_override"], {"num_predict": 32})

    def test_runtime_tuning_controls_apply_through_real_ipc_service_path(self) -> None:
        captured: dict[str, object] = {}

        class _CapturingRuntime:
            backend_name = "simulation"

            def load_model(
                self,
                model_name: str,
                *,
                runtime_profile: str | None = None,
                backend_options: dict[str, object] | None = None,
            ):
                from astrawave.inference_runtime import InferenceModelBinding

                captured["load_profile"] = runtime_profile
                captured["load_options"] = dict(backend_options or {})
                return InferenceModelBinding(
                    backend="simulation",
                    requested_model_name=model_name,
                    resolved_model_name=model_name,
                    metadata={},
                )

            def generate(
                self,
                model_name: str,
                *,
                prompt: str,
                step_name: str,
                max_tokens: int | None = None,
                temperature: float | None = None,
                system_prompt: str | None = None,
                backend_options: dict[str, object] | None = None,
            ) -> dict[str, object]:
                captured["step_options"] = dict(backend_options or {})
                return {
                    "ok": True,
                    "backend": "simulation",
                    "model_name": model_name,
                    "output_text": "ok",
                }

        service = AstraWeaveService(
            inference_runtime_factory=lambda backend_name: _CapturingRuntime(),
            security_guard=SecurityGuard(SecurityPolicy(service_owner_sid=self.owner.user_sid)),
        )
        server = AstraWeaveIpcServer(
            service=service,
            prefer_named_pipe=False,
            host="127.0.0.1",
            port=0,
        )
        server.start()
        self.addCleanup(server.stop)
        endpoint = _endpoint_to_uri(server.endpoint)
        client = _connect_client(endpoint, self.owner)
        self.addCleanup(client.close)

        session_id = client.CreateSession(self.owner)
        client.LoadModel(
            session_id,
            "demo-model",
            self.owner,
            runtime_profile="vram_constrained",
            runtime_backend_options={"num_ctx": 4096, "num_batch": 2},
        )
        client.RunStep(
            session_id,
            "decode",
            self.owner,
            prompt="hello",
            runtime_profile_override="memory_saver",
            runtime_backend_options_override={"num_ctx": 1024},
        )

        self.assertEqual(captured["load_profile"], "vram_constrained")
        load_options = captured["load_options"]
        self.assertIsInstance(load_options, dict)
        self.assertEqual(load_options["num_ctx"], 4096)
        self.assertEqual(load_options["num_batch"], 2)
        self.assertTrue(load_options["low_vram"])
        step_options = captured["step_options"]
        self.assertIsInstance(step_options, dict)
        self.assertEqual(step_options["num_ctx"], 1024)
        self.assertTrue(step_options["low_vram"])

    def test_remote_errors_propagate_as_typed_api_errors(self) -> None:
        sdk = self._make_sdk(self.owner)
        session_id = sdk.CreateSession()
        sdk.close()

        foreign_sdk = self._make_sdk(self.foreign)
        with self.assertRaises(ApiError) as cm:
            foreign_sdk.GetResidency(session_id)
        self.assertEqual(cm.exception.code, ApiErrorCode.AUTH_DENIED)
        self.assertIn("AW_ERR_AUTH_DENIED", str(cm.exception))
        foreign_sdk.close()

        direct_client = self._make_client(self.owner)
        invalid_session_id = direct_client.CreateSession(self.owner)
        self.addCleanup(lambda: direct_client.CloseSession(invalid_session_id, self.owner))

        with self.assertRaises(ApiError) as cm:
            direct_client.RunStep(invalid_session_id, "decode", self.owner)
        self.assertEqual(cm.exception.code, ApiErrorCode.INVALID_STATE)

    def test_connection_rejects_caller_identity_switch_after_binding(self) -> None:
        client = self._make_client(self.owner)
        session_id = client.CreateSession(self.owner)
        self.addCleanup(lambda: client.CloseSession(session_id, self.owner))

        switched = CallerIdentity(user_sid=self.owner.user_sid, pid=self.owner.pid + 99)
        with self.assertRaises(ApiError) as cm:
            client.GetResidency(session_id, switched)
        self.assertEqual(cm.exception.code, ApiErrorCode.AUTH_DENIED)


if __name__ == "__main__":
    unittest.main()
