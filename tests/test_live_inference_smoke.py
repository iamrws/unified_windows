"""Contract tests for the live inference smoke workflow."""

from __future__ import annotations

import unittest

from scripts.live_inference_smoke import run_live_inference_smoke


class LiveInferenceSmokeTests(unittest.TestCase):
    def test_run_live_inference_smoke_simulation_backend_returns_structured_json(self) -> None:
        result = run_live_inference_smoke(
            model_name="demo-model",
            prompt="Say hello in one short sentence.",
            runtime_backend="simulation",
            service_runstep_mode="simulation",
            max_tokens=32,
            temperature=0.2,
        )

        self.assertEqual(result["model_name"], "demo-model")
        self.assertEqual(result["runtime_backend"], "simulation")
        self.assertEqual(result["service"]["load_model"]["runtime_backend"], "simulation")
        self.assertEqual(result["service"]["run_step"]["step_name"], "prompt_smoke")

        inference = result["inference"]["result"]
        self.assertEqual(inference["backend"], "simulation")
        self.assertEqual(inference["model_name"], "demo-model")
        self.assertTrue(inference["ok"])
        self.assertIn("output_text", inference)

    def test_run_live_inference_smoke_auto_profiles_large_models(self) -> None:
        result = run_live_inference_smoke(
            model_name="qwen2.5:14b",
            prompt="Summarize the effect of large host RAM on local inference.",
            runtime_backend="simulation",
            runtime_backend_options={"num_ctx": 4096},
            runtime_profile_override="memory_saver",
            runtime_backend_options_override={"num_predict": 64, "top_p": 0.9},
            service_runstep_mode="simulation",
            max_tokens=32,
            temperature=0.2,
        )

        tuning = result["runtime_tuning"]
        self.assertEqual(tuning["model_size_billion"], 14.0)
        self.assertEqual(tuning["effective"]["load_profile"], "vram_constrained")
        self.assertEqual(tuning["effective"]["load_backend_options"]["num_ctx"], 4096)
        self.assertEqual(tuning["effective"]["step_profile_override"], "memory_saver")
        self.assertEqual(tuning["effective"]["step_backend_options_override"]["num_predict"], 64)
        self.assertEqual(result["service"]["load_model"]["runtime_backend_options"]["num_ctx"], 4096)
        self.assertEqual(result["service"]["run_step"]["runtime_profile_override"], "memory_saver")
        self.assertEqual(result["service"]["run_step"]["runtime_backend_options_override"]["num_predict"], 64)
        self.assertEqual(result["inference"]["runtime_profile"], "memory_saver")
        self.assertEqual(result["inference"]["runtime_backend_options"]["num_predict"], 64)


if __name__ == "__main__":
    unittest.main()
