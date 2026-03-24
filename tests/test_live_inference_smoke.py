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


if __name__ == "__main__":
    unittest.main()
