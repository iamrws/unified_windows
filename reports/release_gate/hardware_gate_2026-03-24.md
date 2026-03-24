# Hardware Gate Report

- Run id: `2026-03-24`
- Generated at: `2026-03-24T19:27:38.459147+00:00`
- Verdict: `pass`

| Gate | Result | Note |
| --- | --- | --- |
| `service_hardware_mode_active` | `pass` | run_mode=hardware |
| `service_hardware_transfer_ok` | `pass` | hardware_result.ok=True |
| `service_triggered_nvml_delta_observed` | `pass` | observed_delta_bytes=172556288 |

## Runtime Result

```json
{
  "correlation_id": "7c806411-ef14-4557-8a12-6509c44559bc:hardware_gate:run-90400d45-8d93-4816-8e90-c59ff1a57481",
  "fallback_result": null,
  "fallback_step": null,
  "hardware_result": {
    "cuda_symbols": {
      "ctx_create": "cuCtxCreate_v2",
      "ctx_destroy": "cuCtxDestroy_v2",
      "mem_alloc": "cuMemAlloc_v2",
      "mem_free": "cuMemFree_v2",
      "memcpy_d2h": "cuMemcpyDtoH_v2",
      "memcpy_h2d": "cuMemcpyHtoD_v2"
    },
    "device_index": 0,
    "error": null,
    "nvml_observation": {
      "after_alloc": {
        "device_index": 0,
        "error": null,
        "source": "nvml",
        "timestamp_ms": 1774380458040,
        "used_bytes": 2414010368
      },
      "after_copy": {
        "device_index": 0,
        "error": null,
        "source": "nvml",
        "timestamp_ms": 1774380458126,
        "used_bytes": 2414010368
      },
      "after_free": {
        "device_index": 0,
        "error": null,
        "source": "nvml",
        "timestamp_ms": 1774380458240,
        "used_bytes": 2241060864
      },
      "before": {
        "device_index": 0,
        "error": null,
        "source": "nvml",
        "timestamp_ms": 1774380457359,
        "used_bytes": 2241454080
      },
      "device_index": 0,
      "observed": true,
      "observed_delta_bytes": 172556288,
      "peak_used_bytes": 2414010368,
      "source": "nvml"
    },
    "ok": true,
    "service_context": {
      "device_index": 0,
      "hold_ms": 300,
      "requested_transfer_bytes": 67108864,
      "session_id": "7c806411-ef14-4557-8a12-6509c44559bc",
      "step_name": "hardware_gate"
    },
    "size_bytes": 67108864,
    "throughput": {
      "round_trip_gib_per_sec": 10.394491750931346
    },
    "timestamp_ms": 1774380458375,
    "timings_ms": {
      "copy_d2h_ms": 6.4788,
      "copy_h2d_ms": 5.5468,
      "total_ms": 1016.1867
    },
    "verification": {
      "dst_crc32": "f43b1a25",
      "matched": true,
      "src_crc32": "f43b1a25"
    }
  },
  "policy_profile": "stability",
  "pressure_level": 0.0,
  "run_mode": "hardware",
  "session_id": "7c806411-ef14-4557-8a12-6509c44559bc",
  "state": "READY",
  "step_name": "hardware_gate"
}
```
