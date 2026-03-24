# Threat Model Sign-Off

- Run id: `2026-03-24`
- Generated at: `2026-03-24T19:28:29.549151+00:00`
- Verdict: `pass`

| Threat | Result | Evidence |
| --- | --- | --- |
| `T-001` | `pass` | `test_security_contract.SecurityContractTests.test_unknown_or_cross_user_caller_is_denied_by_default`, `test_ipc_server.IpcServerContractTests.test_server_requires_explicit_caller_identity_by_default`, `test_ipc_client_sdk.IpcClientSdkE2ETests.test_connection_rejects_caller_identity_switch_after_binding` |
| `T-002` | `pass` | `test_integrated_service_contract.IntegratedServiceContractTests.test_session_ownership_isolation_denies_cross_user_access`, `test_phase3_runtime_smoke.Phase3RuntimeSmokeTests.test_remote_call_smoke_path_denies_foreign_caller` |
| `T-003` | `pass` | `test_ipc_protocol.IpcProtocolContractTests.test_request_envelope_validation_rejects_malformed_input`, `test_ipc_protocol.IpcProtocolContractTests.test_request_envelope_rejects_payloads_over_max_size`, `test_ipc_server.IpcServerContractTests.test_server_rejects_oversized_request_payload` |
| `T-005` | `pass` | `test_telemetry_contract.TelemetryContractTests.test_local_only_is_default_and_export_is_disabled`, `test_telemetry_contract.TelemetryContractTests.test_export_requires_explicit_opt_in` |

- Source unittest log: `C:\Users\grzeg\Documents\Jito\testmarch24\unified_windows\reports\runlogs\unittest_rc_2026-03-24.txt`
