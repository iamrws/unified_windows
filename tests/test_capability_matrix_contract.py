"""Compatibility-matrix capability mode contract tests."""

from __future__ import annotations

import unittest

from astrawave.capabilities import CapabilityMode, CapabilitySignals, resolve_capability_mode


class CapabilityMatrixContractTests(unittest.TestCase):
    def test_profile_pa_primary_maps_to_numa_dgpu(self) -> None:
        snapshot = resolve_capability_mode(
            CapabilitySignals(
                supports_runtime=True,
                has_discrete_gpu=True,
                is_uma=False,
                is_cache_coherent_uma=False,
            )
        )
        self.assertEqual(snapshot.mode, CapabilityMode.NUMA_DGPU)
        self.assertEqual(snapshot.reason_code, "CAPABILITY_NUMA_DGPU")

    def test_profile_pb_secondary_maps_to_cache_coherent_uma(self) -> None:
        snapshot = resolve_capability_mode(
            CapabilitySignals(
                supports_runtime=True,
                has_discrete_gpu=False,
                is_uma=True,
                is_cache_coherent_uma=True,
            )
        )
        self.assertEqual(snapshot.mode, CapabilityMode.CACHE_COHERENT_UMA)
        self.assertEqual(snapshot.reason_code, "CAPABILITY_CACHE_COHERENT_UMA")

    def test_profile_pc_degraded_maps_to_uma(self) -> None:
        snapshot = resolve_capability_mode(
            CapabilitySignals(
                supports_runtime=True,
                has_discrete_gpu=False,
                is_uma=True,
                is_cache_coherent_uma=False,
            )
        )
        self.assertEqual(snapshot.mode, CapabilityMode.UMA)
        self.assertEqual(snapshot.reason_code, "CAPABILITY_UMA")

    def test_profile_pd_unsupported_fails_fast(self) -> None:
        snapshot = resolve_capability_mode(
            CapabilitySignals(
                supports_runtime=False,
                has_discrete_gpu=False,
                is_uma=False,
                is_cache_coherent_uma=False,
            )
        )
        self.assertEqual(snapshot.mode, CapabilityMode.UNSUPPORTED)
        self.assertEqual(snapshot.reason_code, "CAPABILITY_UNSUPPORTED_RUNTIME")


if __name__ == "__main__":
    unittest.main()
