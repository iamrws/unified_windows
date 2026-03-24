# Compatibility Matrix Evidence

- Run id: `2026-03-24`
- Generated at: `2026-03-24T20:36:54.998968+00:00`
- Verdict: `pass`

| Profile | Test | Result |
| --- | --- | --- |
| `P-A` | `test_capability_matrix_contract.CapabilityMatrixContractTests.test_profile_pa_primary_maps_to_numa_dgpu` | `pass` |
| `P-B` | `test_capability_matrix_contract.CapabilityMatrixContractTests.test_profile_pb_secondary_maps_to_cache_coherent_uma` | `pass` |
| `P-C` | `test_capability_matrix_contract.CapabilityMatrixContractTests.test_profile_pc_degraded_maps_to_uma` | `pass` |
| `P-D` | `test_capability_matrix_contract.CapabilityMatrixContractTests.test_profile_pd_unsupported_fails_fast` | `pass` |

- Notes:
  - `P-A` evidence uses real RC workload runs.
  - `P-B/P-C/P-D` evidence is branch-contract validation via capability mapping tests.
