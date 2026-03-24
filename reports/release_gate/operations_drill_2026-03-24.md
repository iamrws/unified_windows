# Operations Drill Report

- Run id: `2026-03-24`
- Started: `2026-03-24T14:16:43.492876+00:00`
- Ended: `2026-03-24T14:16:43.500345+00:00`
- Verdict: `pass`

| Drill | Owner | Result | Expected | Observed |
| --- | --- | --- | --- | --- |
| `DRILL-AUTH-ABUSE` | Security owner | `pass` | Foreign caller is denied when attempting session access | Foreign caller denied with AW_ERR_AUTH_DENIED as expected |
| `DRILL-BUDGET-CONTRACTION` | Platform owner | `pass` | Synthetic budget contraction raises pressure and emits reason-coded fallback; restored budget lowers pressure without hard failure | Budget contraction elevated pressure and triggered reason-coded fallback; pressure dropped after restoring budget |
| `DRILL-ROLLBACK` | Release owner | `pass` | Rollback path serves successful probe traffic after candidate shutdown | Candidate runtime served traffic, was stopped, and replacement runtime served traffic |

## Drill Details

### DRILL-AUTH-ABUSE - Unauthorized caller rejection

- Started: `2026-03-24T14:16:43.492885+00:00`
- Ended: `2026-03-24T14:16:43.493154+00:00`
```json
{
  "error_code": "AW_ERR_AUTH_DENIED",
  "error_message": "AW_ERR_AUTH_DENIED: caller is not authorized for this local service",
  "session_id": "bd105d80-22b1-45aa-b964-b6846cdd867b"
}
```

### DRILL-BUDGET-CONTRACTION - Budget contraction and recovery

- Started: `2026-03-24T14:16:43.493164+00:00`
- Ended: `2026-03-24T14:16:43.493633+00:00`
```json
{
  "contracted_budget_bytes": 16777216,
  "correlation_id": "81ac52c8-38ee-4d89-a48d-c025384e9c7d:decode:run-bedd63de-402f-4fef-990e-4027aa0331f8",
  "fallback_reason_code": "FALLBACK_INITIAL_STEP",
  "original_budget_bytes": 8589934592,
  "pressure_before": 0.0078125,
  "pressure_contracted": 1.0,
  "pressure_recovered": 0.0,
  "session_id": "81ac52c8-38ee-4d89-a48d-c025384e9c7d"
}
```

### DRILL-ROLLBACK - Candidate-to-previous rollback validation

- Started: `2026-03-24T14:16:43.493643+00:00`
- Ended: `2026-03-24T14:16:43.500326+00:00`
```json
{
  "candidate_endpoint": "tcp://127.0.0.1:52618",
  "candidate_served_requests": 3,
  "rollback_endpoint": "tcp://127.0.0.1:52620",
  "rollback_served_requests": 3
}
```
