"""Contract tests for the core AstraWeave service lifecycle."""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from astrawave import AstraWeaveService
from astrawave.errors import ApiError, ApiErrorCode
from astrawave.types import MemoryTier


class AstraWeaveServiceLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = AstraWeaveService(runstep_mode="simulation")
        self.session_id = self.service.CreateSession()

    def assertApiErrorCode(self, exc: Exception, expected: ApiErrorCode) -> None:
        self.assertIsInstance(exc, ApiError)
        self.assertEqual(exc.code, expected)

    def test_invalid_order_rejection_blocks_out_of_sequence_calls(self) -> None:
        with self.assertRaises(ApiError) as cm:
            self.service.RunStep(self.session_id)
        self.assertApiErrorCode(cm.exception, ApiErrorCode.INVALID_STATE)

        with self.assertRaises(ApiError) as cm:
            self.service.RegisterTensor(self.session_id, "kv", 1024)
        self.assertApiErrorCode(cm.exception, ApiErrorCode.INVALID_STATE)

        with self.assertRaises(ApiError) as cm:
            self.service.SetTierHint(self.session_id, "kv", MemoryTier.HOT)
        self.assertApiErrorCode(cm.exception, ApiErrorCode.INVALID_STATE)

        with self.assertRaises(ApiError) as cm:
            self.service.PrefetchPlan(self.session_id)
        self.assertApiErrorCode(cm.exception, ApiErrorCode.INVALID_STATE)

        self.service.LoadModel(self.session_id, "demo-model")
        self.service.RegisterTensor(self.session_id, "kv", 1024)
        self.service.SetTierHint(self.session_id, "kv", MemoryTier.WARM)

        result = self.service.RunStep(self.session_id, step_name="decode")
        self.assertEqual(result["session_id"], self.session_id)
        self.assertEqual(result["step_name"], "decode")

    def test_close_session_is_idempotent(self) -> None:
        self.service.LoadModel(self.session_id, "demo-model")
        self.service.CloseSession(self.session_id)

        # Second close must be a no-op, not an exception.
        self.service.CloseSession(self.session_id)

        with self.assertRaises(ApiError) as cm:
            self.service.LoadModel(self.session_id, "another-model")
        self.assertIn(cm.exception.code, {ApiErrorCode.INVALID_STATE, ApiErrorCode.NOT_FOUND})

    def test_register_tensor_after_load_is_allowed(self) -> None:
        self.service.LoadModel(self.session_id, "demo-model")
        self.service.RegisterTensor(self.session_id, "weights", 4096)

        residency = self.service.GetResidency(self.session_id)
        self.assertEqual(residency.session_id, self.session_id)
        self.assertGreaterEqual(residency.pinned_ram_bytes, 4096)

    def test_run_step_rejects_second_active_execution(self) -> None:
        self.service.LoadModel(self.session_id, "demo-model")
        self.service.RegisterTensor(self.session_id, "kv", 1024)
        session = self.service._get_session(self.session_id)  # contract-level active-run guard check
        with session.lock:
            session.active_run = True
        try:
            with self.assertRaises(ApiError) as cm:
                self.service.RunStep(self.session_id, step_name="decode")
            self.assertApiErrorCode(cm.exception, ApiErrorCode.CONFLICT_RUN_IN_PROGRESS)
        finally:
            with session.lock:
                session.active_run = False

    def test_run_step_hardware_mode_invokes_executor_and_returns_hardware_payload(self) -> None:
        calls: list[dict[str, int]] = []

        def fake_executor(*, size_bytes: int, device_index: int, hold_ms: int) -> dict[str, object]:
            calls.append(
                {
                    "size_bytes": size_bytes,
                    "device_index": device_index,
                    "hold_ms": hold_ms,
                }
            )
            return {
                "ok": True,
                "nvml_observation": {
                    "observed": True,
                    "observed_delta_bytes": 4096,
                },
            }

        service = AstraWeaveService(runstep_mode="hardware", hardware_executor=fake_executor)
        session_id = service.CreateSession()
        service.LoadModel(session_id, "demo-model")
        service.RegisterTensor(session_id, "kv", 1024)

        result = service.RunStep(session_id, step_name="decode")
        self.assertEqual(result["requested_run_mode"], "hardware")
        self.assertEqual(result["run_mode"], "hardware")
        self.assertIsInstance(result["hardware_result"], dict)
        self.assertTrue(result["hardware_result"]["ok"])
        self.assertEqual(len(calls), 1)
        self.assertGreaterEqual(calls[0]["size_bytes"], 1024)

    def test_run_step_hardware_mode_marks_session_degraded_on_runtime_failure(self) -> None:
        def failing_executor(*, size_bytes: int, device_index: int, hold_ms: int) -> dict[str, object]:
            return {
                "ok": False,
                "error": {"code": "CUDA_POC_DRIVER_MISSING", "message": "nvcuda.dll was not found"},
            }

        service = AstraWeaveService(runstep_mode="hardware", hardware_executor=failing_executor)
        session_id = service.CreateSession()
        service.LoadModel(session_id, "demo-model")
        service.RegisterTensor(session_id, "kv", 2048)

        result = service.RunStep(session_id, step_name="decode")
        self.assertEqual(result["requested_run_mode"], "hardware")
        self.assertEqual(result["run_mode"], "simulation")
        self.assertIsInstance(result["hardware_result"], dict)
        self.assertFalse(result["hardware_result"]["ok"])
        self.assertTrue(result["hardware_result"]["fallback_to_simulation"])
        self.assertEqual(result["state"].value, "DEGRADED")

    def test_env_flag_can_enable_hardware_mode_without_constructor_override(self) -> None:
        def fake_executor(*, size_bytes: int, device_index: int, hold_ms: int) -> dict[str, object]:
            return {"ok": True}

        with patch.dict(os.environ, {"ASTRAWEAVE_ENABLE_HARDWARE_RUNSTEP": "1"}, clear=False):
            service = AstraWeaveService(hardware_executor=fake_executor)
            session_id = service.CreateSession()
            service.LoadModel(session_id, "demo-model")
            service.RegisterTensor(session_id, "kv", 4096)
            result = service.RunStep(session_id, step_name="decode")
            self.assertEqual(result["requested_run_mode"], "hardware")
            self.assertEqual(result["run_mode"], "hardware")

    def test_auto_mode_prefers_hardware_and_falls_back_cleanly(self) -> None:
        def failing_executor(*, size_bytes: int, device_index: int, hold_ms: int) -> dict[str, object]:
            return {
                "ok": False,
                "error": {"code": "CUDA_POC_DRIVER_MISSING", "message": "nvcuda.dll was not found"},
            }

        service = AstraWeaveService(hardware_executor=failing_executor)
        session_id = service.CreateSession()
        service.LoadModel(session_id, "demo-model")
        service.RegisterTensor(session_id, "kv", 2048)

        result = service.RunStep(session_id, step_name="decode")
        self.assertEqual(result["requested_run_mode"], "auto")
        self.assertEqual(result["run_mode"], "simulation")
        self.assertFalse(result["hardware_result"]["ok"])
        self.assertTrue(result["hardware_result"]["fallback_to_simulation"])
        self.assertEqual(result["state"].value, "DEGRADED")

    def test_run_step_with_prompt_uses_simulation_inference_without_telemetry_leakage(self) -> None:
        self.service.LoadModel(self.session_id, "demo-model")
        result = self.service.RunStep(
            self.session_id,
            step_name="decode",
            prompt="Tell me a secret about penguins.",
            max_tokens=32,
            temperature=0.1,
        )

        self.assertIn("inference_result", result)
        inference = result["inference_result"]
        self.assertTrue(inference["ok"])
        self.assertEqual(inference["backend"], "simulation")
        self.assertEqual(inference["model_name"], "demo-model")
        self.assertIn("output_text", inference)
        telemetry_blob = json.dumps([record.to_dict() for record in self.service.telemetry.records], sort_keys=True)
        self.assertNotIn("Tell me a secret about penguins.", telemetry_blob)

    def test_load_model_with_ollama_prefix_routes_to_injected_runtime(self) -> None:
        calls: list[dict[str, object]] = []

        class FakeOllamaRuntime:
            backend_name = "ollama"

            def load_model(self, model_name: str):
                calls.append({"phase": "load_model", "model_name": model_name})
                from astrawave.inference_runtime import InferenceModelBinding

                return InferenceModelBinding(
                    backend="ollama",
                    requested_model_name=model_name,
                    resolved_model_name=model_name,
                    metadata={"transport": "fake-http"},
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
            ) -> dict[str, object]:
                calls.append(
                    {
                        "phase": "generate",
                        "model_name": model_name,
                        "prompt": prompt,
                        "step_name": step_name,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "system_prompt": system_prompt,
                    }
                )
                return {
                    "ok": True,
                    "backend": "ollama",
                    "model_name": model_name,
                    "output_text": "hello from ollama",
                    "finish_reason": "stop",
                }

        def runtime_factory(backend_name: str):
            if backend_name == "ollama":
                return FakeOllamaRuntime()
            raise AssertionError(f"unexpected backend requested: {backend_name}")

        service = AstraWeaveService(
            runstep_mode="simulation",
            inference_runtime_factory=runtime_factory,
        )
        session_id = service.CreateSession()
        service.LoadModel(session_id, "ollama:qwen2.5:7b")
        result = service.RunStep(
            session_id,
            step_name="decode",
            prompt="Write one short line.",
            max_tokens=24,
            temperature=0.3,
        )

        session = service._get_session(session_id)
        self.assertEqual(session.inference_backend, "ollama")
        self.assertEqual(session.resolved_model_name, "qwen2.5:7b")
        self.assertEqual(calls[0], {"phase": "load_model", "model_name": "qwen2.5:7b"})
        self.assertEqual(calls[1]["phase"], "generate")
        self.assertEqual(calls[1]["model_name"], "qwen2.5:7b")
        self.assertEqual(calls[1]["step_name"], "decode")
        self.assertEqual(result["inference_result"]["backend"], "ollama")
        self.assertEqual(result["inference_result"]["output_text"], "hello from ollama")

    def test_load_model_runtime_backend_override_routes_without_prefix(self) -> None:
        class FakeOllamaRuntime:
            backend_name = "ollama"

            def load_model(self, model_name: str):
                from astrawave.inference_runtime import InferenceModelBinding

                return InferenceModelBinding(
                    backend="ollama",
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
            ) -> dict[str, object]:
                return {
                    "ok": True,
                    "backend": "ollama",
                    "model_name": model_name,
                    "output_text": f"mock:{prompt}",
                }

        def runtime_factory(backend_name: str):
            if backend_name == "ollama":
                return FakeOllamaRuntime()
            raise AssertionError(f"unexpected backend requested: {backend_name}")

        service = AstraWeaveService(runstep_mode="simulation", inference_runtime_factory=runtime_factory)
        session_id = service.CreateSession()
        service.LoadModel(session_id, "qwen2.5:7b", runtime_backend="ollama")
        result = service.RunStep(session_id, step_name="decode", prompt="hello")

        session = service._get_session(session_id)
        self.assertEqual(session.inference_backend, "ollama")
        self.assertEqual(session.resolved_model_name, "qwen2.5:7b")
        self.assertEqual(result["inference_result"]["backend"], "ollama")

    def test_large_model_load_applies_vram_constrained_profile_and_merges_prompt_options(self) -> None:
        calls: list[dict[str, object]] = []

        class FakeOllamaRuntime:
            backend_name = "ollama"

            def load_model(
                self,
                model_name: str,
                *,
                runtime_profile: str | None = None,
                backend_options: dict[str, object] | None = None,
            ):
                calls.append(
                    {
                        "phase": "load_model",
                        "model_name": model_name,
                        "runtime_profile": runtime_profile,
                        "backend_options": backend_options,
                    }
                )
                from astrawave.inference_runtime import InferenceModelBinding

                return InferenceModelBinding(
                    backend="ollama",
                    requested_model_name=model_name,
                    resolved_model_name=model_name,
                    metadata={"transport": "fake-http"},
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
                calls.append(
                    {
                        "phase": "generate",
                        "model_name": model_name,
                        "prompt": prompt,
                        "step_name": step_name,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "system_prompt": system_prompt,
                        "backend_options": backend_options,
                    }
                )
                return {
                    "ok": True,
                    "backend": "ollama",
                    "model_name": model_name,
                    "output_text": "large model response",
                    "finish_reason": "stop",
                }

        def runtime_factory(backend_name: str):
            if backend_name == "ollama":
                return FakeOllamaRuntime()
            raise AssertionError(f"unexpected backend requested: {backend_name}")

        service = AstraWeaveService(
            runstep_mode="simulation",
            inference_runtime_factory=runtime_factory,
        )
        session_id = service.CreateSession()
        service.LoadModel(session_id, "ollama:llama3:50b")

        session = service._get_session(session_id)
        self.assertEqual(session.inference_metadata["runtime_profile"], "vram_constrained")
        self.assertEqual(session.inference_metadata["model_size_billion"], 50.0)
        self.assertTrue(session.inference_metadata["backend_options"]["low_vram"])

        self.assertEqual(calls[0]["phase"], "load_model")
        self.assertEqual(calls[0]["runtime_profile"], "vram_constrained")
        self.assertTrue(calls[0]["backend_options"]["low_vram"])
        self.assertEqual(calls[0]["backend_options"]["num_ctx"], 1536)

        result = service.RunStep(
            session_id,
            step_name="decode",
            prompt="Write one short line.",
            max_tokens=24,
            backend_options={"num_ctx": 3072, "repeat_penalty": 1.05},
        )

        self.assertEqual(result["inference_result"]["backend"], "ollama")
        self.assertEqual(calls[1]["phase"], "generate")
        self.assertEqual(calls[1]["model_name"], "llama3:50b")
        self.assertEqual(calls[1]["max_tokens"], 24)
        self.assertEqual(calls[1]["backend_options"]["num_ctx"], 3072)
        self.assertEqual(calls[1]["backend_options"]["num_batch"], 16)
        self.assertTrue(calls[1]["backend_options"]["low_vram"])
        self.assertAlmostEqual(calls[1]["backend_options"]["repeat_penalty"], 1.05)

    def test_run_step_runtime_profile_override_applies_constrained_defaults(self) -> None:
        calls: list[dict[str, object]] = []

        class FakeOllamaRuntime:
            backend_name = "ollama"

            def load_model(
                self,
                model_name: str,
                *,
                runtime_profile: str | None = None,
                backend_options: dict[str, object] | None = None,
            ):
                calls.append(
                    {
                        "phase": "load_model",
                        "model_name": model_name,
                        "runtime_profile": runtime_profile,
                        "backend_options": backend_options,
                    }
                )
                from astrawave.inference_runtime import InferenceModelBinding

                return InferenceModelBinding(
                    backend="ollama",
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
                calls.append(
                    {
                        "phase": "generate",
                        "model_name": model_name,
                        "step_name": step_name,
                        "backend_options": backend_options,
                    }
                )
                return {
                    "ok": True,
                    "backend": "ollama",
                    "model_name": model_name,
                    "output_text": "override-response",
                }

        def runtime_factory(backend_name: str):
            if backend_name == "ollama":
                return FakeOllamaRuntime()
            raise AssertionError(f"unexpected backend requested: {backend_name}")

        service = AstraWeaveService(runstep_mode="simulation", inference_runtime_factory=runtime_factory)
        session_id = service.CreateSession()
        service.LoadModel(session_id, "qwen2.5:7b", runtime_backend="ollama")
        service.RunStep(
            session_id,
            step_name="decode",
            prompt="hello",
            runtime_profile_override="memory_saver",
            runtime_backend_options_override={"num_ctx": 1024},
        )

        self.assertEqual(calls[0]["phase"], "load_model")
        self.assertEqual(calls[0]["runtime_profile"], "default")
        self.assertEqual(calls[1]["phase"], "generate")
        self.assertEqual(calls[1]["backend_options"]["num_ctx"], 1024)
        self.assertTrue(calls[1]["backend_options"]["low_vram"])


if __name__ == "__main__":
    unittest.main()
