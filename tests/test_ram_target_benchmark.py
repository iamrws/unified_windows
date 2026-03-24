"""Contract tests for RAM-target benchmark/autotune script."""

from __future__ import annotations

import unittest

from scripts.ram_target_benchmark import (
    CandidateSummary,
    _extract_metrics,
    build_default_candidates,
    select_best_candidate,
)


class RamTargetBenchmarkContractTests(unittest.TestCase):
    def test_build_default_candidates_for_70b_class_prefers_fit_profiles(self) -> None:
        candidates = build_default_candidates(72.0)
        ids = [item.candidate_id for item in candidates]
        self.assertEqual(ids[0], "q4_fit")
        self.assertIn("q4_balance", ids)
        self.assertIn("q6_push", ids)
        self.assertEqual(candidates[0].runtime_options["num_ctx"], 2048)

    def test_extract_metrics_computes_decode_and_end_to_end_throughput(self) -> None:
        smoke_result = {
            "inference": {
                "result": {
                    "usage": {
                        "eval_count": 120,
                        "eval_duration": 6_000_000_000,
                        "total_duration": 10_000_000_000,
                        "load_duration": 1_000_000_000,
                        "prompt_eval_count": 40,
                        "prompt_eval_duration": 2_000_000_000,
                    },
                    "finish_reason": "stop",
                }
            }
        }
        metrics = _extract_metrics(smoke_result)
        self.assertAlmostEqual(metrics["eval_tokens_per_second"], 20.0)
        self.assertAlmostEqual(metrics["end_to_end_tokens_per_second"], 12.0)
        self.assertAlmostEqual(metrics["total_duration_seconds"], 10.0)
        self.assertEqual(metrics["finish_reason"], "stop")

    def test_select_best_candidate_prefers_ram_target_proximity_then_speed(self) -> None:
        summaries = [
            CandidateSummary(
                candidate_id="A",
                compression_hint="q4",
                notes="a",
                runtime_options={},
                runs=[],
                success_count=1,
                run_count=1,
                success_rate=1.0,
                avg_eval_tokens_per_second=9.0,
                avg_end_to_end_tokens_per_second=5.0,
                avg_total_seconds=20.0,
                peak_private_gb=70.0,
                peak_working_set_gb=72.0,
                ram_target_delta_gb=-20.0,
            ),
            CandidateSummary(
                candidate_id="B",
                compression_hint="q6",
                notes="b",
                runtime_options={},
                runs=[],
                success_count=1,
                run_count=1,
                success_rate=1.0,
                avg_eval_tokens_per_second=7.0,
                avg_end_to_end_tokens_per_second=4.0,
                avg_total_seconds=25.0,
                peak_private_gb=88.0,
                peak_working_set_gb=90.0,
                ram_target_delta_gb=-2.0,
            ),
            CandidateSummary(
                candidate_id="C",
                compression_hint="q8",
                notes="c",
                runtime_options={},
                runs=[],
                success_count=1,
                run_count=1,
                success_rate=1.0,
                avg_eval_tokens_per_second=8.5,
                avg_end_to_end_tokens_per_second=4.8,
                avg_total_seconds=22.0,
                peak_private_gb=88.0,
                peak_working_set_gb=90.0,
                ram_target_delta_gb=-2.0,
            ),
        ]
        best = select_best_candidate(summaries, target_ram_gb=90.0)
        self.assertIsNotNone(best)
        assert best is not None
        # B and C are equally close to target RAM, so throughput tie-breaker picks C.
        self.assertEqual(best.candidate_id, "C")


if __name__ == "__main__":
    unittest.main()
