"""Contract tests for runtime tuning and backend option synthesis."""

from __future__ import annotations

import unittest

from astrawave.errors import ApiError, ApiErrorCode
from astrawave.inference_runtime import OllamaInferenceRuntime
from astrawave.runtime_tuning import infer_model_size_billion, resolve_runtime_tuning


class RuntimeTuningContractTests(unittest.TestCase):
    def test_infer_model_size_billion_detects_simple_and_moe_tags(self) -> None:
        self.assertEqual(infer_model_size_billion("qwen2.5:14b-instruct"), 14.0)
        self.assertEqual(infer_model_size_billion("mixtral-8x7b-instruct"), 56.0)
        self.assertIsNone(infer_model_size_billion("demo-model"))

    def test_large_models_default_to_vram_constrained_profile(self) -> None:
        tuning = resolve_runtime_tuning("llama3:50b")

        self.assertEqual(tuning.profile_name, "vram_constrained")
        self.assertEqual(tuning.model_size_billion, 50.0)
        self.assertEqual(tuning.model_size_label, "50b")
        self.assertEqual(tuning.backend_options["num_ctx"], 1536)
        self.assertEqual(tuning.backend_options["num_batch"], 16)
        self.assertTrue(tuning.backend_options["low_vram"])
        self.assertTrue(tuning.backend_options["f16_kv"])

    def test_explicit_backend_options_override_profile_defaults(self) -> None:
        tuning = resolve_runtime_tuning(
            "llama3:50b",
            backend_options={"num_ctx": 4096, "repeat_penalty": 1.1},
        )

        self.assertEqual(tuning.profile_name, "vram_constrained")
        self.assertEqual(tuning.backend_options["num_ctx"], 4096)
        self.assertEqual(tuning.backend_options["num_batch"], 16)
        self.assertTrue(tuning.backend_options["low_vram"])
        self.assertAlmostEqual(tuning.backend_options["repeat_penalty"], 1.1)

    def test_invalid_backend_option_types_are_rejected(self) -> None:
        with self.assertRaises(ApiError) as cm:
            resolve_runtime_tuning("qwen2.5:14b", backend_options={"num_ctx": object()})
        self.assertEqual(cm.exception.code, ApiErrorCode.INVALID_ARGUMENT)

    def test_ollama_runtime_merges_backend_options_before_transport(self) -> None:
        requests: list[dict[str, object]] = []

        def fake_transport(url: str, payload: dict[str, object], timeout_seconds: float) -> dict[str, object]:
            requests.append(
                {
                    "url": url,
                    "payload": payload,
                    "timeout_seconds": timeout_seconds,
                }
            )
            return {
                "response": "hello from ollama",
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 11,
                "eval_count": 24,
                "total_duration": 42,
                "load_duration": 7,
                "eval_duration": 35,
            }

        runtime = OllamaInferenceRuntime(base_url="http://127.0.0.1:11434", timeout_seconds=2.5, transport=fake_transport)
        binding = runtime.load_model("llama3:50b")
        effective_options = {
            **binding.metadata["backend_options"],
            "num_ctx": 4096,
            "repeat_penalty": 1.05,
        }
        result = runtime.generate(
            binding.resolved_model_name,
            prompt="Explain the fit.",
            step_name="prompt_smoke",
            max_tokens=32,
            temperature=0.2,
            backend_options=effective_options,
        )

        self.assertEqual(binding.metadata["runtime_profile"], "vram_constrained")
        self.assertEqual(binding.metadata["backend_options"]["num_ctx"], 1536)
        self.assertEqual(requests[0]["url"], "http://127.0.0.1:11434/api/generate")
        self.assertEqual(requests[0]["timeout_seconds"], 2.5)
        payload = requests[0]["payload"]
        self.assertEqual(payload["model"], "llama3:50b")
        self.assertEqual(payload["prompt"], "Explain the fit.")
        self.assertEqual(payload["options"]["num_ctx"], 4096)
        self.assertEqual(payload["options"]["num_batch"], 16)
        self.assertTrue(payload["options"]["low_vram"])
        self.assertTrue(payload["options"]["f16_kv"])
        self.assertEqual(payload["options"]["num_predict"], 32)
        self.assertAlmostEqual(payload["options"]["temperature"], 0.2)
        self.assertAlmostEqual(payload["options"]["repeat_penalty"], 1.05)
        self.assertTrue(result["ok"])
        self.assertEqual(result["output_text"], "hello from ollama")


if __name__ == "__main__":
    unittest.main()
