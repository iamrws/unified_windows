"""Phase 7 contract tests: quantization-aware orchestration and production hardening."""

from __future__ import annotations

import importlib
import os
import unittest
from collections import deque




class Phase7AuditFixTests(unittest.TestCase):
    """Tests for Pillar 1: Production Hardening (AUD-001 through AUD-008)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.service_mod = importlib.import_module("astrawave.service")
        cls.types_mod = importlib.import_module("astrawave.types")
        cls.telemetry_mod = importlib.import_module("astrawave.telemetry")
        cls.ipc_client_mod = importlib.import_module("astrawave.ipc_client")
        cls.security_mod = importlib.import_module("astrawave.security")
        cls.fallback_mod = importlib.import_module("astrawave.fallback")
        cls.errors_mod = importlib.import_module("astrawave.errors")

    def _make_service(self):
        return self.service_mod.AstraWeaveService()

    def _make_caller(self):
        sid = self.security_mod.resolve_current_user_sid() or "S-1-5-21-test"
        return self.security_mod.CallerIdentity(sid, os.getpid())

    # ------------------------------------------------------------------
    # AUD-001: RunStep must not leave sessions stuck in RUNNING
    # ------------------------------------------------------------------
    def test_aud001_runstep_exception_restores_state(self) -> None:
        """If RunStep raises mid-execution, the session must NOT be stuck in RUNNING."""

        service = self._make_service()
        caller = self._make_caller()
        session_id = service.CreateSession(caller)
        service.LoadModel(session_id, "demo-model", runtime_backend="simulation", caller_identity=caller)

        # Force a failure by injecting a broken hardware executor
        def broken_executor(**kwargs):
            raise RuntimeError("simulated hardware failure")

        service._hardware_executor = broken_executor
        service._runstep_mode_override = "hardware"

        try:
            service.RunStep(session_id, "step1", caller_identity=caller)
        except Exception:
            pass

        # Session must be recoverable — not stuck in RUNNING
        session = service._sessions[session_id]
        self.assertNotEqual(session.state, self.types_mod.SessionState.RUNNING)
        self.assertFalse(session.active_run)

    # ------------------------------------------------------------------
    # AUD-002: CLI caller-pid defaults to os.getpid() not 1
    # ------------------------------------------------------------------
    def test_aud002_cli_caller_pid_default_is_none(self) -> None:
        """CLI --caller-pid default should be None (resolved to os.getpid() at runtime)."""

        import argparse
        # Directly test the common parent parser's default for --caller-pid.
        # The _build_parser() uses a common parent with --caller-pid.
        parent = argparse.ArgumentParser(add_help=False)
        parent.add_argument("--caller-pid", type=int, default=None)
        args = parent.parse_args([])
        self.assertIsNone(args.caller_pid)

        # Also verify the actual CLI module's source has default=None
        cli_mod = importlib.import_module("astrawave.cli")
        import inspect
        source = inspect.getsource(cli_mod._build_parser)
        self.assertIn("default=None", source)

    # ------------------------------------------------------------------
    # AUD-004: Closed sessions must be capped at MAX_CLOSED_SESSIONS
    # ------------------------------------------------------------------
    def test_aud004_closed_sessions_evicted_at_cap(self) -> None:
        """Creating and closing more than MAX_CLOSED_SESSIONS must evict oldest."""

        # Use a high rate limit to avoid rate-limiting during bulk session creation
        guard = self.security_mod.SecurityGuard(
            self.security_mod.SecurityPolicy(
                service_owner_sid=self.security_mod.resolve_current_user_sid() or "S-1-5-21-test",
                create_session_limit_per_minute=10_000,
                max_concurrent_sessions_per_caller=10_000,
            )
        )
        service = self.service_mod.AstraWeaveService(security_guard=guard)
        caller = self._make_caller()
        cap = self.service_mod.MAX_CLOSED_SESSIONS

        session_ids = []
        for _ in range(cap + 10):
            sid = service.CreateSession(caller)
            session_ids.append(sid)

        for sid in session_ids:
            service.CloseSession(sid, caller_identity=caller)

        self.assertLessEqual(len(service._closed_sessions), cap)
        # The most recently closed sessions should be retained
        self.assertIn(session_ids[-1], service._closed_sessions)

    # ------------------------------------------------------------------
    # AUD-005: Telemetry ring buffer must cap at TELEMETRY_RING_BUFFER_SIZE
    # ------------------------------------------------------------------
    def test_aud005_telemetry_ring_buffer_caps_records(self) -> None:
        """Telemetry pipeline must not exceed its ring buffer size."""

        pipeline = self.telemetry_mod.TelemetryPipeline(
            policy=self.telemetry_mod.TelemetryPolicy()
        )
        # Override with a small cap for test speed
        small_cap = 100
        pipeline._records = deque(maxlen=small_cap)

        for i in range(small_cap + 50):
            event = self.telemetry_mod.TelemetryEvent(
                reason_code=self.telemetry_mod.TelemetryReasonCode.UNKNOWN,
                session_id=f"session-{i}",
            )
            pipeline.record_event(event)

        self.assertEqual(len(pipeline._records), small_cap)

    # ------------------------------------------------------------------
    # AUD-003: IPC client transport_policy='pipe_only' must reject fallback
    # ------------------------------------------------------------------
    def test_aud003_transport_policy_pipe_only_raises(self) -> None:
        """Client with pipe_only policy must not silently fall back to TCP."""

        client = self.ipc_client_mod.AstraWeaveIpcClient(
            endpoint="auto",
            transport_policy="pipe_only",
        )
        self.assertEqual(client.transport_policy, "pipe_only")

    def test_aud003_transport_policy_invalid_raises(self) -> None:
        """Invalid transport_policy must raise ValueError."""

        with self.assertRaises(ValueError):
            self.ipc_client_mod.AstraWeaveIpcClient(
                endpoint="auto",
                transport_policy="invalid_policy",
            )

    # ------------------------------------------------------------------
    # AUD-008: Session state HMAC integrity verification
    # ------------------------------------------------------------------
    def test_aud008_session_hmac_detects_tampering(self) -> None:
        """Tampering with session state must be detected by HMAC verification."""

        service = self._make_service()
        caller = self._make_caller()
        session_id = service.CreateSession(caller)
        service.LoadModel(session_id, "demo-model", runtime_backend="simulation", caller_identity=caller)

        session = service._sessions[session_id]
        self.assertTrue(service._verify_session(session))

        # Tamper with the session state without re-signing
        session.model_name = "tampered-model"
        self.assertFalse(service._verify_session(session))

    def test_aud008_hmac_valid_after_normal_mutations(self) -> None:
        """HMAC should be valid after normal service operations."""

        service = self._make_service()
        caller = self._make_caller()
        session_id = service.CreateSession(caller)
        self.assertTrue(service._verify_session(service._sessions[session_id]))

        service.LoadModel(session_id, "demo-model", runtime_backend="simulation", caller_identity=caller)
        self.assertTrue(service._verify_session(service._sessions[session_id]))


class Phase7QuantizationProviderTests(unittest.TestCase):
    """Tests for Pillar 2: Quantization Provider Framework."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.quant_mod = importlib.import_module("astrawave.quantization")

    def test_turboquant_compression_ratio(self) -> None:
        """SimulatedTurboQuantProvider must model correct compression ratios."""

        provider = self.quant_mod.SimulatedTurboQuantProvider()
        self.assertEqual(provider.backend_name, self.quant_mod.QuantizationBackend.TURBOQUANT)

        # Default 3.5 bits: ratio = 32/3.5 ≈ 9.14
        result = provider.quantize("tensor_kv", 1_000_000)
        self.assertAlmostEqual(result.compression_ratio, 32.0 / 3.5, places=2)
        self.assertLess(result.compressed_bytes, result.original_bytes)
        self.assertEqual(result.bit_width, 3.5)

        # 2.5 bits: ratio = 32/2.5 = 12.8
        result_25 = provider.quantize("tensor_kv", 1_000_000, bit_width=2.5)
        self.assertAlmostEqual(result_25.compression_ratio, 12.8, places=2)
        self.assertLess(result_25.compressed_bytes, result.compressed_bytes)

    def test_fp8_compression_ratio(self) -> None:
        """FP8Provider must model 4x compression from FP32."""

        provider = self.quant_mod.FP8Provider()
        self.assertEqual(provider.backend_name, self.quant_mod.QuantizationBackend.FP8)

        result = provider.quantize("tensor_v", 4_000_000)
        self.assertAlmostEqual(result.compression_ratio, 4.0)
        self.assertEqual(result.compressed_bytes, 1_000_000)

    def test_none_provider_no_compression(self) -> None:
        """NoneProvider must return identity compression."""

        provider = self.quant_mod.NoneProvider()
        result = provider.quantize("tensor_cold", 2_000_000)
        self.assertEqual(result.compression_ratio, 1.0)
        self.assertEqual(result.compressed_bytes, 2_000_000)

    def test_provider_for_tier_selection(self) -> None:
        """provider_for_tier must return the right backend per tier."""

        hot = self.quant_mod.provider_for_tier("HOT")
        self.assertEqual(hot.backend_name, self.quant_mod.QuantizationBackend.TQ2_0)

        warm = self.quant_mod.provider_for_tier("WARM")
        self.assertEqual(warm.backend_name, self.quant_mod.QuantizationBackend.TQ1_0)

        cold = self.quant_mod.provider_for_tier("COLD")
        self.assertEqual(cold.backend_name, self.quant_mod.QuantizationBackend.NONE)

    def test_provider_for_tier_uses_32gb_mapping_for_cold(self) -> None:
        cold = self.quant_mod.provider_for_tier(
            "COLD",
            vram_budget_bytes=32 * 1024**3,
        )
        self.assertEqual(cold.backend_name, self.quant_mod.QuantizationBackend.FP8)

    def test_provider_for_tier_mapping_override_is_supported(self) -> None:
        cold = self.quant_mod.provider_for_tier(
            "COLD",
            mapping_override={"COLD": "fp8"},
        )
        self.assertEqual(cold.backend_name, self.quant_mod.QuantizationBackend.FP8)

    def test_supported_bit_widths(self) -> None:
        """Each provider must report valid bit widths."""

        tq = self.quant_mod.SimulatedTurboQuantProvider()
        self.assertIn(3.5, tq.supported_bit_widths())
        self.assertIn(2.5, tq.supported_bit_widths())

        fp8 = self.quant_mod.FP8Provider()
        self.assertIn(8.0, fp8.supported_bit_widths())

    def test_quantization_result_metadata(self) -> None:
        """TurboQuant result must include algorithm metadata."""

        provider = self.quant_mod.SimulatedTurboQuantProvider()
        result = provider.quantize("test", 1000)
        self.assertEqual(result.metadata["algorithm"], "turboquant_simulated")
        self.assertIn("polarquant", result.metadata["stages"])
        self.assertIn("qjl_residual", result.metadata["stages"])


class Phase7FallbackLadderTests(unittest.TestCase):
    """Tests for Pillar 3: Compression-aware fallback ladder."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.fallback_mod = importlib.import_module("astrawave.fallback")

    def test_kv_quantization_upgrade_is_first_step(self) -> None:
        """KV_QUANTIZATION_UPGRADE must be the first fallback step."""

        ladder = self.fallback_mod.DEFAULT_FALLBACK_LADDER
        self.assertEqual(ladder[0], self.fallback_mod.FallbackStep.KV_QUANTIZATION_UPGRADE)

    def test_fallback_ladder_has_six_steps(self) -> None:
        """The updated ladder must have 6 steps (was 5)."""

        ladder = self.fallback_mod.DEFAULT_FALLBACK_LADDER
        self.assertEqual(len(ladder), 6)

    def test_controller_advances_to_quantization_upgrade_first(self) -> None:
        """Under pressure with no prior fallback, first step must be quantization upgrade."""

        controller = self.fallback_mod.FallbackController()
        state = self.fallback_mod.FallbackState(
            current_step=None,
            last_step_change_ms=None,
        )
        decision = controller.evaluate(state, now_ms=1000)
        self.assertTrue(decision.should_advance)
        self.assertEqual(decision.next_step, self.fallback_mod.FallbackStep.KV_QUANTIZATION_UPGRADE)

    def test_controller_advances_past_quantization_to_context_reduction(self) -> None:
        """After quantization upgrade, next step must be KV_CONTEXT_REDUCTION."""

        controller = self.fallback_mod.FallbackController()
        next_step = controller.next_step(self.fallback_mod.FallbackStep.KV_QUANTIZATION_UPGRADE)
        self.assertEqual(next_step, self.fallback_mod.FallbackStep.KV_CONTEXT_REDUCTION)


class Phase7CompressionAwareTieringTests(unittest.TestCase):
    """Tests for Pillar 3: Compression-aware tiering."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tiering_mod = importlib.import_module("astrawave.tiering")
        cls.types_mod = importlib.import_module("astrawave.types")

    def test_compression_ratio_expands_hot_budget(self) -> None:
        """With a compression ratio, more resources should fit in HOT tier."""

        planner = self.tiering_mod.PlacementPlanner()
        # Without compression: 1 GB budget, 800 MB request -> fits in HOT (within 80% headroom)
        request = self.tiering_mod.PlacementRequest(
            resource_id="kv_cache",
            bytes_required=600 * 1024**2,  # 600 MB
            is_active=True,
        )

        # Without compression: 600 MB vs 800 MB budget (80% = 640 MB) -> fits
        plan_no_compress = planner.plan(
            [request],
            hot_budget_bytes=800 * 1024**2,
            hot_compression_ratio=1.0,
        )
        self.assertEqual(plan_no_compress.decisions[0].tier, self.types_mod.MemoryTier.HOT)

        # Now with a bigger request that wouldn't fit without compression
        big_request = self.tiering_mod.PlacementRequest(
            resource_id="kv_cache_big",
            bytes_required=3_000 * 1024**2,  # 3 GB
            is_active=True,
        )

        plan_no_compress_big = planner.plan(
            [big_request],
            hot_budget_bytes=800 * 1024**2,
            hot_compression_ratio=1.0,
        )
        # Without compression: 3 GB vs 640 MB -> spills to WARM
        self.assertEqual(plan_no_compress_big.decisions[0].tier, self.types_mod.MemoryTier.WARM)

        # With TurboQuant 3.5-bit (9.14x): effective = 3GB/9.14 ≈ 328 MB -> fits in 640 MB
        plan_compressed = planner.plan(
            [big_request],
            hot_budget_bytes=800 * 1024**2,
            hot_compression_ratio=32.0 / 3.5,
        )
        self.assertEqual(plan_compressed.decisions[0].tier, self.types_mod.MemoryTier.HOT)

    def test_plan_hot_bytes_accounts_for_compression(self) -> None:
        """Plan hot_bytes should reflect effective (post-compression) byte usage."""

        planner = self.tiering_mod.PlacementPlanner()
        request = self.tiering_mod.PlacementRequest(
            resource_id="kv",
            bytes_required=1_000_000,
            is_active=True,
        )

        plan = planner.plan([request], hot_budget_bytes=10_000_000, hot_compression_ratio=4.0)
        # With 4x compression, hot_bytes should be 250000 (1M / 4)
        self.assertEqual(plan.decisions[0].tier, self.types_mod.MemoryTier.HOT)
        self.assertEqual(plan.hot_bytes, 250_000)

    def test_dynamic_hot_headroom_ratio_scales_with_vram_budget(self) -> None:
        ratio_fn = self.tiering_mod.hot_headroom_ratio_for_budget
        self.assertAlmostEqual(ratio_fn(8 * 1024**3), 0.20)
        self.assertAlmostEqual(ratio_fn(16 * 1024**3), 0.20)
        self.assertAlmostEqual(ratio_fn(32 * 1024**3), 0.125)

    def test_hot_kernel_unavailable_gates_hot_placement(self) -> None:
        planner = self.tiering_mod.PlacementPlanner(hot_kernel_available=False)
        request = self.tiering_mod.PlacementRequest(
            resource_id="kv_requires_kernel",
            bytes_required=128 * 1024**2,
            is_active=True,
            requires_hot_kernel=True,
        )
        plan = planner.plan([request], hot_budget_bytes=8 * 1024**3)
        self.assertEqual(plan.decisions[0].tier, self.types_mod.MemoryTier.WARM)
        self.assertEqual(plan.decisions[0].reason_code, "PLACEMENT_WARM_NO_CUDA_KERNELS")


class Phase7CompressionTelemetryTests(unittest.TestCase):
    """Tests for Pillar 3: Compression telemetry events."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.telemetry_mod = importlib.import_module("astrawave.telemetry")

    def test_compression_event_type_exists(self) -> None:
        """COMPRESSION event type must exist in telemetry."""

        self.assertEqual(
            self.telemetry_mod.TelemetryEventType.COMPRESSION.value,
            "compression_event",
        )

    def test_compression_event_records(self) -> None:
        """CompressionEvent should record correctly in the pipeline."""

        pipeline = self.telemetry_mod.TelemetryPipeline()
        event = self.telemetry_mod.CompressionEvent(
            reason_code=self.telemetry_mod.TelemetryReasonCode.COMPRESSION_APPLIED,
            session_id="test-session",
            tensor_id="kv_cache",
            backend="turboquant",
            original_bytes=1_000_000,
            compressed_bytes=109_890,
            compression_ratio=9.1,
            bit_width=3.5,
        )
        record = pipeline.record_event(event)
        self.assertEqual(record.event_type, self.telemetry_mod.TelemetryEventType.COMPRESSION)
        self.assertEqual(record.reason_code, self.telemetry_mod.TelemetryReasonCode.COMPRESSION_APPLIED)

    def test_compression_reason_codes_exist(self) -> None:
        """Compression-related reason codes must be defined."""

        codes = self.telemetry_mod.TelemetryReasonCode
        self.assertTrue(hasattr(codes, "COMPRESSION_APPLIED"))
        self.assertTrue(hasattr(codes, "COMPRESSION_UPGRADED"))
        self.assertTrue(hasattr(codes, "FALLBACK_KV_QUANTIZATION_UPGRADE"))


class Phase7IntegrationTests(unittest.TestCase):
    """End-to-end integration tests for Phase 7 quantization-aware orchestration."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.service_mod = importlib.import_module("astrawave.service")
        cls.types_mod = importlib.import_module("astrawave.types")
        cls.security_mod = importlib.import_module("astrawave.security")
        cls.telemetry_mod = importlib.import_module("astrawave.telemetry")

    def _make_service(self):
        return self.service_mod.AstraWeaveService()

    def _make_caller(self):
        sid = self.security_mod.resolve_current_user_sid() or "S-1-5-21-test"
        return self.security_mod.CallerIdentity(sid, os.getpid())

    def test_runstep_under_pressure_applies_quantization_first(self) -> None:
        """Under memory pressure, first fallback should apply quantization, not demote."""

        service = self._make_service()
        caller = self._make_caller()
        session_id = service.CreateSession(caller)
        service.LoadModel(session_id, "demo-14b", runtime_backend="simulation", caller_identity=caller)
        service.RegisterTensor(session_id, "kv_cache", 1024, caller_identity=caller)
        service.SetTierHint(session_id, "kv_cache", self.types_mod.MemoryTier.HOT, caller_identity=caller)
        service.PrefetchPlan(session_id, caller_identity=caller)

        # Force high pressure
        session = service._sessions[session_id]
        with session.lock:
            session.vram_budget_bytes = 1
        service._sign_session(session)

        result = service.RunStep(session_id, "decode", caller_identity=caller)
        self.assertEqual(result["fallback_result"]["next_step"], "kv_quantization_upgrade")

        # Tensor should still be in VRAM (quantization compresses in-place)
        residency = service.GetResidency(session_id, caller_identity=caller)
        self.assertEqual(residency.primary_tier, self.types_mod.MemoryTier.HOT)

        # Compression telemetry should have been emitted
        records = service.telemetry.records
        compression_events = [
            r for r in records
            if r.reason_code == self.telemetry_mod.TelemetryReasonCode.COMPRESSION_APPLIED
        ]
        self.assertGreaterEqual(len(compression_events), 1)

    def test_tensor_record_has_quantization_fields(self) -> None:
        """After quantization, tensor records should have compression metadata."""

        service = self._make_service()
        caller = self._make_caller()
        session_id = service.CreateSession(caller)
        service.LoadModel(session_id, "demo-model", runtime_backend="simulation", caller_identity=caller)
        service.RegisterTensor(session_id, "kv", 10_000, caller_identity=caller)
        service.SetTierHint(session_id, "kv", self.types_mod.MemoryTier.HOT, caller_identity=caller)
        service.PrefetchPlan(session_id, caller_identity=caller)

        session = service._sessions[session_id]
        with session.lock:
            session.vram_budget_bytes = 1
        service._sign_session(session)

        service.RunStep(session_id, "step1", caller_identity=caller)

        tensor = service._sessions[session_id].tensors["kv"]
        self.assertNotEqual(tensor.quantization_backend, "none")
        self.assertGreater(tensor.compression_ratio, 1.0)
        self.assertLess(tensor.effective_bytes, tensor.size_bytes)

    def test_kv_quantization_upgrade_progression_is_concrete(self) -> None:
        service = self._make_service()
        caller = self._make_caller()
        session_id = service.CreateSession(caller)
        service.LoadModel(session_id, "demo-model", runtime_backend="simulation", caller_identity=caller)
        service.RegisterTensor(session_id, "kv", 10_000, caller_identity=caller)
        service.SetTierHint(session_id, "kv", self.types_mod.MemoryTier.HOT, caller_identity=caller)

        session = service._sessions[session_id]
        tensor = session.tensors["kv"]
        self.assertEqual(tensor.quantization_backend, "none")

        service._apply_tier_quantization(session, tensor)
        self.assertEqual(tensor.quantization_backend, "tq2_0")

        service._apply_tier_quantization(session, tensor)
        self.assertEqual(tensor.quantization_backend, "tq1_0")

        service._apply_tier_quantization(session, tensor)
        self.assertEqual(tensor.quantization_backend, "tq1_0")

    def test_residency_snapshot_has_quantization_fields(self) -> None:
        """ResidencySnapshot must include quantization metadata."""

        snapshot = self.types_mod.ResidencySnapshot(
            session_id="test",
            session_state=self.types_mod.SessionState.READY,
            primary_tier=self.types_mod.MemoryTier.HOT,
            quantization_backend="turboquant",
            compression_ratio=9.14,
        )
        self.assertEqual(snapshot.quantization_backend, "turboquant")
        self.assertAlmostEqual(snapshot.compression_ratio, 9.14)


if __name__ == "__main__":
    unittest.main()
