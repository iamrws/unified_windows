# 07 - Executive Summary

## Severity Scorecard

| Severity | Count | Finding IDs |
| --- | --- | --- |
| `CRITICAL` | 3 | `AUD-001`, `AUD-002`, `AUD-007` |
| `HIGH` | 4 | `AUD-003`, `AUD-004`, `AUD-005`, `AUD-006` |
| `MEDIUM` | 1 | `AUD-008` |
| **Total** | **8** | |

## Summary of Findings

| ID | Severity | Title | Status |
| --- | --- | --- | --- |
| `AUD-001` | `CRITICAL` | `RunStep` leaves sessions stuck in `RUNNING` after argument validation error | Open |
| `AUD-002` | `CRITICAL` | CLI defaults break remote auth on healthy server (`caller-pid=1`) | Open |
| `AUD-003` | `HIGH` | IPC client silently downgrades from named pipe to TCP `127.0.0.1:8765` | Open |
| `AUD-004` | `HIGH` | Closed sessions accumulate without bound (`_closed_sessions`) | Open |
| `AUD-005` | `HIGH` | Telemetry records accumulate without bound (`TelemetryPipeline._records`) | Open |
| `AUD-006` | `HIGH` | Test system is non-hermetic; default `pytest` path is broken | Open |
| `AUD-007` | `CRITICAL` | No reproducible build/lint gate is defined for production release | Open |
| `AUD-008` | `MEDIUM` | Local simulator state stored in predictable temp file without integrity checks | Open |

## Prioritized Action Plan

### Immediate (Sprint 1)
1. **AUD-007** - Add `pyproject.toml` with build, test, lint, and type-check configuration. Enforce as CI gate.
2. **AUD-001** - Fix `RunStep` state machine: validate inputs before transitioning to `RUNNING`; add rollback on validation failure. Add regression tests.
3. **AUD-002** - Default CLI caller PID to `os.getpid()` and SID to `resolve_current_user_sid()`. Keep override flags for explicit impersonation tests only.

### Short-term (Sprint 2)
4. **AUD-006** - Package project for editable install, add `pytest.ini` or `pyproject.toml` test config, remove `PYTHONPATH` injection hacks.
5. **AUD-004** - Replace full closed-session retention with capped tombstones (TTL/LRU). Add configurable hard cap.
6. **AUD-005** - Enforce max in-memory telemetry record count. Add periodic cleanup in `record_event` or via background scheduler.

### Medium-term (Sprint 3)
7. **AUD-003** - Make pipe-to-TCP transport fallback explicit and opt-in. Emit typed downgrade telemetry.
8. **AUD-008** - Add file integrity MAC/signature and restrictive ACLs to simulator state files.

## Risk Assessment

- **Operational risk**: `AUD-001` can permanently block user sessions with a single bad request. This is the highest-impact user-facing bug.
- **Security risk**: `AUD-002` makes remote mode unusable out-of-the-box; `AUD-003` silently downgrades transport security.
- **Scalability risk**: `AUD-004` and `AUD-005` create unbounded memory growth that will cause OOM in long-running deployments.
- **Process risk**: `AUD-007` means there is no automated quality gate preventing regressions from shipping.

## Audit Metadata
- **Date**: 2026-03-24
- **Scope**: Full production audit of `unified_windows` repository
- **Tests executed**: 97/97 passed (unittest), 97/97 passed (pytest with workaround)
- **Static analysis**: `compileall` passed; no lint/type-check config present

## Confidence
`[HIGH]`
