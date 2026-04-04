"""Tests for TurboQuant CUDA integration features (F-010 through F-033).

Covers: TQ1_0/TQ2_0 providers, tier-provider mapping, KV cache options,
throughput profile, dynamic headroom, quantization-aware placement,
KV quantization progression, and throughput fallback timings.
"""

from __future__ import annotations

import importlib
import unittest


class TQ10ProviderTests(unittest.TestCase):
    """F-011: TQ1_0Provider with real GGML block compression ratios."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.quant = importlib.import_module("astrawave.quantization")

    def test_backend_name(self) -> None:
        provider = self.quant.TQ1_0Provider()
        self.assertEqual(provider.backend_name, self.quant.QuantizationBackend.TQ1_0)

    def test_compression_ratio(self) -> None:
        provider = self.quant.TQ1_0Provider()
        ratio = provider.estimate_compression_ratio(1_000_000)
        self.assertAlmostEqual(ratio, 32.0 / 1.6875, places=2)

    def test_bit_width(self) -> None:
        provider = self.quant.TQ1_0Provider()
        self.assertEqual(provider.supported_bit_widths(), (1.6875,))

    def test_quantize_result(self) -> None:
        provider = self.quant.TQ1_0Provider()
        result = provider.quantize("kv_cache", 1_000_000)
        self.assertEqual(result.backend, self.quant.QuantizationBackend.TQ1_0)
        self.assertEqual(result.original_bytes, 1_000_000)
        self.assertLess(result.compressed_bytes, result.original_bytes)
        self.assertAlmostEqual(result.bit_width, 1.6875)
        self.assertEqual(result.metadata["ggml_type_id"], 34)
        self.assertEqual(result.metadata["block_size"], 256)
        self.assertEqual(result.metadata["block_bytes"], 42)
        self.assertEqual(result.metadata["encoding"], "ternary_base3")

    def test_minimum_compressed_bytes(self) -> None:
        provider = self.quant.TQ1_0Provider()
        result = provider.quantize("tiny", 1)
        self.assertGreaterEqual(result.compressed_bytes, 1)


class TQ20ProviderTests(unittest.TestCase):
    """F-011: TQ2_0Provider with real GGML block compression ratios."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.quant = importlib.import_module("astrawave.quantization")

    def test_backend_name(self) -> None:
        provider = self.quant.TQ2_0Provider()
        self.assertEqual(provider.backend_name, self.quant.QuantizationBackend.TQ2_0)

    def test_compression_ratio(self) -> None:
        provider = self.quant.TQ2_0Provider()
        ratio = provider.estimate_compression_ratio(1_000_000)
        self.assertAlmostEqual(ratio, 32.0 / 2.0625, places=2)

    def test_bit_width(self) -> None:
        provider = self.quant.TQ2_0Provider()
        self.assertEqual(provider.supported_bit_widths(), (2.0625,))

    def test_quantize_result(self) -> None:
        provider = self.quant.TQ2_0Provider()
        result = provider.quantize("kv_cache", 1_000_000)
        self.assertEqual(result.backend, self.quant.QuantizationBackend.TQ2_0)
        self.assertEqual(result.metadata["ggml_type_id"], 35)
        self.assertEqual(result.metadata["block_size"], 256)
        self.assertEqual(result.metadata["block_bytes"], 66)
        self.assertEqual(result.metadata["encoding"], "2bit_polar")

    def test_tq2_less_compression_than_tq1(self) -> None:
        """TQ2_0 should have lower compression ratio than TQ1_0."""
        tq1 = self.quant.TQ1_0Provider()
        tq2 = self.quant.TQ2_0Provider()
        self.assertGreater(
            tq1.estimate_compression_ratio(1000),
            tq2.estimate_compression_ratio(1000),
        )


class TierProviderMappingTests(unittest.TestCase):
    """F-012: Configurable tier-provider mapping with 32GB VRAM support."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.quant = importlib.import_module("astrawave.quantization")

    def test_default_profile_backward_compatible(self) -> None:
        """Default profile should match original 8GB mapping."""
        hot = self.quant.provider_for_tier("HOT")
        self.assertEqual(hot.backend_name, self.quant.QuantizationBackend.TURBOQUANT)
        warm = self.quant.provider_for_tier("WARM")
        self.assertEqual(warm.backend_name, self.quant.QuantizationBackend.FP8)
        cold = self.quant.provider_for_tier("COLD")
        self.assertEqual(cold.backend_name, self.quant.QuantizationBackend.NONE)

    def test_high_vram_profile(self) -> None:
        """32GB VRAM profile uses TQ2_0 for HOT, TQ1_0 for WARM, FP8 for COLD."""
        hot = self.quant.provider_for_tier("HOT", profile="high_vram")
        self.assertEqual(hot.backend_name, self.quant.QuantizationBackend.TQ2_0)
        warm = self.quant.provider_for_tier("WARM", profile="high_vram")
        self.assertEqual(warm.backend_name, self.quant.QuantizationBackend.TQ1_0)
        cold = self.quant.provider_for_tier("COLD", profile="high_vram")
        self.assertEqual(cold.backend_name, self.quant.QuantizationBackend.FP8)

    def test_custom_mapping_overrides(self) -> None:
        custom = {"HOT": self.quant.FP8Provider, "WARM": self.quant.NoneProvider}
        hot = self.quant.provider_for_tier("HOT", custom_mapping=custom)
        self.assertEqual(hot.backend_name, self.quant.QuantizationBackend.FP8)

    def test_unknown_tier_falls_back_to_none(self) -> None:
        provider = self.quant.provider_for_tier("UNKNOWN")
        self.assertEqual(provider.backend_name, self.quant.QuantizationBackend.NONE)

    def test_case_insensitive(self) -> None:
        hot = self.quant.provider_for_tier("hot", profile="high_vram")
        self.assertEqual(hot.backend_name, self.quant.QuantizationBackend.TQ2_0)


class KVCacheOptionTests(unittest.TestCase):
    """F-013/F-014: KV cache type and flash_attn validation."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tuning = importlib.import_module("astrawave.runtime_tuning")
        cls.errors = importlib.import_module("astrawave.errors")

    def test_valid_kv_types_accepted(self) -> None:
        for kv_type in ("f16", "f32", "q8_0", "q4_0", "tq1_0", "tq2_0"):
            self.tuning.validate_kv_cache_options({"type_k": kv_type})

    def test_invalid_type_k_rejected(self) -> None:
        with self.assertRaises(self.errors.ApiError):
            self.tuning.validate_kv_cache_options({"type_k": "q3_bogus"})

    def test_invalid_type_v_rejected(self) -> None:
        with self.assertRaises(self.errors.ApiError):
            self.tuning.validate_kv_cache_options({"type_v": "invalid"})

    def test_quantized_v_without_flash_attn_rejected(self) -> None:
        """Quantized type_v requires flash_attn enabled (llama.cpp constraint)."""
        with self.assertRaises(self.errors.ApiError):
            self.tuning.validate_kv_cache_options({
                "type_v": "tq2_0",
                "flash_attn": False,
            })

    def test_quantized_v_with_flash_attn_accepted(self) -> None:
        self.tuning.validate_kv_cache_options({
            "type_v": "tq2_0",
            "flash_attn": True,
        })

    def test_f16_v_without_flash_attn_accepted(self) -> None:
        self.tuning.validate_kv_cache_options({
            "type_v": "f16",
            "flash_attn": False,
        })

    def test_no_options_passes(self) -> None:
        self.tuning.validate_kv_cache_options({})


class ThroughputProfileTests(unittest.TestCase):
    """F-020/F-022: Throughput runtime profile with model-size-aware scaling."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tuning = importlib.import_module("astrawave.runtime_tuning")

    def test_throughput_profile_exists(self) -> None:
        self.assertEqual(self.tuning.THROUGHPUT_RUNTIME_PROFILE, "throughput")

    def test_throughput_alias_resolves(self) -> None:
        resolved = self.tuning.normalize_runtime_profile_name("throughput")
        self.assertEqual(resolved, "throughput")
        resolved2 = self.tuning.normalize_runtime_profile_name("high_throughput")
        self.assertEqual(resolved2, "throughput")

    def test_throughput_small_model_options(self) -> None:
        opts = self.tuning.profile_backend_options("throughput", 7.0)
        self.assertEqual(opts["num_ctx"], 32768)
        self.assertEqual(opts["num_batch"], 512)
        self.assertEqual(opts["type_k"], "tq2_0")
        self.assertEqual(opts["type_v"], "f16")
        self.assertEqual(opts["num_gpu"], -1)
        self.assertTrue(opts["offload_kqv"])

    def test_throughput_medium_model_options(self) -> None:
        opts = self.tuning.profile_backend_options("throughput", 14.0)
        self.assertEqual(opts["num_ctx"], 8192)
        self.assertEqual(opts["num_batch"], 256)

    def test_throughput_large_model_options(self) -> None:
        opts = self.tuning.profile_backend_options("throughput", 34.0)
        self.assertEqual(opts["num_ctx"], 4096)
        self.assertEqual(opts["num_batch"], 128)

    def test_throughput_huge_model_uses_tq1(self) -> None:
        opts = self.tuning.profile_backend_options("throughput", 70.0)
        self.assertEqual(opts["num_ctx"], 2048)
        self.assertEqual(opts["num_batch"], 64)
        self.assertEqual(opts["type_k"], "tq1_0")

    def test_resolve_runtime_tuning_throughput(self) -> None:
        tuning = self.tuning.resolve_runtime_tuning(
            "qwen2-7b",
            runtime_profile="throughput",
        )
        self.assertEqual(tuning.profile_name, "throughput")
        self.assertEqual(tuning.backend_options["type_k"], "tq2_0")


class LargeModelThresholdTests(unittest.TestCase):
    """F-021: VRAM-budget-aware large-model threshold."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tuning = importlib.import_module("astrawave.runtime_tuning")

    def test_default_threshold_is_14b(self) -> None:
        self.assertAlmostEqual(self.tuning.large_model_threshold(), 14.0)

    def test_8gb_threshold(self) -> None:
        self.assertAlmostEqual(self.tuning.large_model_threshold(8.0), 14.0)

    def test_32gb_threshold_raises(self) -> None:
        threshold = self.tuning.large_model_threshold(32.0)
        self.assertGreater(threshold, 14.0)
        self.assertLessEqual(threshold, 34.0)

    def test_80gb_threshold_caps_at_70(self) -> None:
        self.assertAlmostEqual(self.tuning.large_model_threshold(80.0), 70.0)

    def test_is_large_model_with_vram(self) -> None:
        # 20B model: large on 8GB, not large on 32GB
        self.assertTrue(self.tuning.is_large_model(20.0, vram_budget_gb=8.0))
        self.assertFalse(self.tuning.is_large_model(20.0, vram_budget_gb=32.0))

    def test_is_large_model_backward_compatible(self) -> None:
        self.assertTrue(self.tuning.is_large_model(14.0))
        self.assertFalse(self.tuning.is_large_model(13.0))
        self.assertFalse(self.tuning.is_large_model(None))


class DynamicHeadroomTests(unittest.TestCase):
    """F-030: Dynamic hot headroom ratio based on VRAM budget."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tiering = importlib.import_module("astrawave.tiering")

    def test_default_headroom_is_20_percent(self) -> None:
        ratio = self.tiering.dynamic_headroom_ratio()
        self.assertAlmostEqual(ratio, 0.20)

    def test_8gb_headroom(self) -> None:
        ratio = self.tiering.dynamic_headroom_ratio(8.0)
        self.assertAlmostEqual(ratio, 0.20, places=2)

    def test_32gb_headroom(self) -> None:
        ratio = self.tiering.dynamic_headroom_ratio(32.0)
        self.assertAlmostEqual(ratio, 0.125)

    def test_80gb_headroom(self) -> None:
        ratio = self.tiering.dynamic_headroom_ratio(80.0)
        self.assertAlmostEqual(ratio, 0.05)

    def test_headroom_clamped_low(self) -> None:
        ratio = self.tiering.dynamic_headroom_ratio(200.0)
        self.assertAlmostEqual(ratio, 0.05)

    def test_policy_for_vram_budget(self) -> None:
        policy = self.tiering.policy_for_vram_budget(32.0)
        self.assertAlmostEqual(policy.hot_headroom_ratio, 0.125)

    def test_policy_for_vram_budget_default(self) -> None:
        policy = self.tiering.policy_for_vram_budget()
        self.assertAlmostEqual(policy.hot_headroom_ratio, 0.20)


class QuantizationAwarePlacementTests(unittest.TestCase):
    """F-031: Placement decisions gated by CUDA kernel availability."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tiering = importlib.import_module("astrawave.tiering")
        cls.types = importlib.import_module("astrawave.types")

    def test_no_cuda_kernels_forces_warm(self) -> None:
        planner = self.tiering.PlacementPlanner()
        request = self.tiering.PlacementRequest(
            resource_id="kv_cache",
            bytes_required=100_000,
            is_active=True,
        )
        decision = planner.classify(
            request,
            hot_budget_bytes=10_000_000,
            hot_compression_ratio=15.52,
            cuda_kernels_available=False,
        )
        self.assertEqual(decision.tier, self.types.MemoryTier.WARM)
        self.assertEqual(decision.reason_code, "PLACEMENT_WARM_NO_CUDA_KERNELS")

    def test_cuda_kernels_available_allows_hot(self) -> None:
        planner = self.tiering.PlacementPlanner()
        request = self.tiering.PlacementRequest(
            resource_id="kv_cache",
            bytes_required=100_000,
            is_active=True,
        )
        decision = planner.classify(
            request,
            hot_budget_bytes=10_000_000,
            hot_compression_ratio=15.52,
            cuda_kernels_available=True,
        )
        self.assertEqual(decision.tier, self.types.MemoryTier.HOT)

    def test_no_compression_ignores_cuda_flag(self) -> None:
        """When compression_ratio=1.0, cuda_kernels_available shouldn't matter."""
        planner = self.tiering.PlacementPlanner()
        request = self.tiering.PlacementRequest(
            resource_id="kv",
            bytes_required=100_000,
            is_active=True,
        )
        decision = planner.classify(
            request,
            hot_budget_bytes=10_000_000,
            hot_compression_ratio=1.0,
            cuda_kernels_available=False,
        )
        self.assertEqual(decision.tier, self.types.MemoryTier.HOT)

    def test_plan_no_cuda_kernels(self) -> None:
        planner = self.tiering.PlacementPlanner()
        requests = [
            self.tiering.PlacementRequest("kv1", 500_000, is_active=True),
            self.tiering.PlacementRequest("kv2", 500_000, is_active=True),
        ]
        plan = planner.plan(
            requests,
            hot_budget_bytes=10_000_000,
            hot_compression_ratio=15.52,
            cuda_kernels_available=False,
        )
        for d in plan.decisions:
            self.assertEqual(d.tier, self.types.MemoryTier.WARM)


class KVQuantizationProgressionTests(unittest.TestCase):
    """F-032: Concrete KV quantization upgrade progression."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.fallback = importlib.import_module("astrawave.fallback")

    def test_progression_has_three_levels(self) -> None:
        self.assertEqual(len(self.fallback.KV_QUANTIZATION_PROGRESSION), 3)

    def test_baseline_is_f16(self) -> None:
        baseline = self.fallback.KV_QUANTIZATION_PROGRESSION[0]
        self.assertEqual(baseline.type_k, "f16")
        self.assertEqual(baseline.type_v, "f16")
        self.assertAlmostEqual(baseline.estimated_compression, 1.0)

    def test_first_upgrade_is_tq2_0(self) -> None:
        level = self.fallback.KV_QUANTIZATION_PROGRESSION[1]
        self.assertEqual(level.type_k, "tq2_0")
        self.assertAlmostEqual(level.estimated_compression, 15.52)

    def test_second_upgrade_is_tq1_0(self) -> None:
        level = self.fallback.KV_QUANTIZATION_PROGRESSION[2]
        self.assertEqual(level.type_k, "tq1_0")
        self.assertAlmostEqual(level.estimated_compression, 18.96)

    def test_next_from_f16(self) -> None:
        nxt = self.fallback.next_kv_quantization_level("f16")
        self.assertIsNotNone(nxt)
        self.assertEqual(nxt.type_k, "tq2_0")

    def test_next_from_tq2_0(self) -> None:
        nxt = self.fallback.next_kv_quantization_level("tq2_0")
        self.assertIsNotNone(nxt)
        self.assertEqual(nxt.type_k, "tq1_0")

    def test_next_from_tq1_0_is_none(self) -> None:
        nxt = self.fallback.next_kv_quantization_level("tq1_0")
        self.assertIsNone(nxt)

    def test_next_from_unknown_is_none(self) -> None:
        nxt = self.fallback.next_kv_quantization_level("q4_0")
        self.assertIsNone(nxt)


class ThroughputFallbackTimingsTests(unittest.TestCase):
    """F-033: Throughput-optimized fallback timings."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.fallback = importlib.import_module("astrawave.fallback")

    def test_throughput_controls_faster(self) -> None:
        tc = self.fallback.THROUGHPUT_OSCILLATION_CONTROLS
        sc = self.fallback.STABILITY_OSCILLATION_CONTROLS
        self.assertLess(tc.cooldown_seconds, sc.cooldown_seconds)
        self.assertLess(tc.minimum_dwell_seconds, sc.minimum_dwell_seconds)
        self.assertLess(tc.churn_window_seconds, sc.churn_window_seconds)

    def test_throughput_controls_values(self) -> None:
        tc = self.fallback.THROUGHPUT_OSCILLATION_CONTROLS
        self.assertEqual(tc.cooldown_seconds, 10)
        self.assertEqual(tc.minimum_dwell_seconds, 5)
        self.assertEqual(tc.churn_window_seconds, 15)

    def test_stability_controls_backward_compatible(self) -> None:
        sc = self.fallback.STABILITY_OSCILLATION_CONTROLS
        self.assertEqual(sc.cooldown_seconds, 30)
        self.assertEqual(sc.minimum_dwell_seconds, 15)
        self.assertEqual(sc.churn_window_seconds, 30)

    def test_controller_with_throughput_controls(self) -> None:
        """Controller should accept throughput timings and advance faster."""
        tc = self.fallback.THROUGHPUT_OSCILLATION_CONTROLS
        controller = self.fallback.FallbackController(controls=tc)

        # After 11 seconds (past 10s throughput cooldown), should advance
        state = self.fallback.FallbackState(
            current_step=self.fallback.FallbackStep.KV_QUANTIZATION_UPGRADE,
            last_step_change_ms=0,
        )
        decision = controller.evaluate(state, now_ms=11_000)
        self.assertTrue(decision.should_advance)

    def test_controller_with_stability_controls_blocks(self) -> None:
        """Stability controller should NOT advance after only 11 seconds."""
        sc = self.fallback.STABILITY_OSCILLATION_CONTROLS
        controller = self.fallback.FallbackController(controls=sc)

        state = self.fallback.FallbackState(
            current_step=self.fallback.FallbackStep.KV_QUANTIZATION_UPGRADE,
            last_step_change_ms=0,
        )
        decision = controller.evaluate(state, now_ms=11_000)
        self.assertFalse(decision.should_advance)


class BackwardCompatibilityTests(unittest.TestCase):
    """Verify existing API surface is not broken."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.quant = importlib.import_module("astrawave.quantization")
        cls.tuning = importlib.import_module("astrawave.runtime_tuning")
        cls.tiering = importlib.import_module("astrawave.tiering")
        cls.fallback = importlib.import_module("astrawave.fallback")

    def test_original_providers_unchanged(self) -> None:
        """SimulatedTurboQuantProvider, FP8Provider, NoneProvider still work."""
        tq = self.quant.SimulatedTurboQuantProvider()
        result = tq.quantize("test", 1_000_000)
        self.assertAlmostEqual(result.compression_ratio, 32.0 / 3.5, places=2)

        fp8 = self.quant.FP8Provider()
        result = fp8.quantize("test", 4_000_000)
        self.assertEqual(result.compressed_bytes, 1_000_000)

        none = self.quant.NoneProvider()
        result = none.quantize("test", 1000)
        self.assertEqual(result.compressed_bytes, 1000)

    def test_provider_for_tier_default_is_backward_compatible(self) -> None:
        hot = self.quant.provider_for_tier("HOT")
        self.assertEqual(hot.backend_name, self.quant.QuantizationBackend.TURBOQUANT)

    def test_vram_constrained_profile_unchanged(self) -> None:
        opts = self.tuning.profile_backend_options("vram_constrained", 14.0)
        self.assertEqual(opts["num_ctx"], 2048)
        self.assertTrue(opts["low_vram"])

    def test_default_fallback_ladder_unchanged(self) -> None:
        ladder = self.fallback.DEFAULT_FALLBACK_LADDER
        self.assertEqual(len(ladder), 6)
        self.assertEqual(ladder[0], self.fallback.FallbackStep.KV_QUANTIZATION_UPGRADE)

    def test_placement_planner_default_behavior(self) -> None:
        planner = self.tiering.PlacementPlanner()
        request = self.tiering.PlacementRequest(
            resource_id="test",
            bytes_required=100_000,
            is_active=True,
        )
        from astrawave.types import MemoryTier
        decision = planner.classify(request, hot_budget_bytes=10_000_000)
        self.assertEqual(decision.tier, MemoryTier.HOT)


if __name__ == "__main__":
    unittest.main()
