# RC Workload Report

- Run id: `2026-03-24`
- Generated at: `2026-03-24T19:28:29.550668+00:00`
- Verdict: `pass`
- W-7B summary: `C:\Users\grzeg\Documents\Jito\testmarch24\unified_windows\reports\runlogs\summary_rc_w7b_chat_2026-03-24.json`
- W-13B summary: `C:\Users\grzeg\Documents\Jito\testmarch24\unified_windows\reports\runlogs\summary_rc_w13b_chat_2026-03-24.json`

| Gate | Result | Details |
| --- | --- | --- |
| Reliability (zero hard OOM proxy via zero failures) | `pass` | 7B failures=0, 13B failures=0 |
| Latency p95 <= 250 ms | `pass` | measured p95=183.845 ms |
| Stability anti-oscillation | `pass` | tests=test_fallback_contract.FallbackContractTests.test_controller_declines_step_changes_inside_cooldown_window, test_fallback_contract.FallbackContractTests.test_controller_respects_cooldown_before_advancing, test_fallback_contract.FallbackContractTests.test_anti_oscillation_defaults_are_locked |
| Explainability reason-code/correlation | `pass` | tests=test_integrated_service_contract.IntegratedServiceContractTests.test_service_emits_telemetry_events_with_stable_reason_codes_and_correlation_ids |
