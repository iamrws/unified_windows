# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.7.x   | Yes       |
| < 0.7   | No        |

## Reporting a Vulnerability

If you discover a security vulnerability in astrawave, please report it responsibly:

1. **Do not** open a public GitHub issue for security vulnerabilities.
2. Email the maintainers at the address listed in the repository profile, or use [GitHub's private vulnerability reporting](https://github.com/iamrws/unified_windows/security/advisories/new).
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Affected versions
   - Suggested fix (if any)

We will acknowledge receipt within 48 hours and aim to release a fix within 7 days for critical issues.

## Security Design

### Zero Dependencies

astrawave intentionally declares **zero runtime dependencies**. The entire package runs on the Python 3.13 standard library plus ctypes FFI to system NVIDIA drivers. This eliminates:

- Dependency confusion attacks
- Transitive dependency compromise
- Typosquatting of upstream packages
- Leftpad-style availability failures

### Local-Only Communication

All IPC is restricted to loopback addresses (127.0.0.1 / named pipes). The IPC server rejects non-loopback bind addresses. The Ollama inference backend validates URL schemes and warns on non-loopback hosts.

### Authentication

The IPC transport uses HMAC-based authkey challenge-response. When no key is configured, an ephemeral key is auto-generated and shared via a restricted-permission file.

### Caller Attestation

On Windows, caller identity is verified via process token SID resolution using a single-handle approach that minimizes TOCTOU risk from PID reuse.

### Known Limitations

- The IPC transport uses Python's `multiprocessing.connection` which serializes via pickle. This is an RCE surface for authenticated peers. Migration to JSON framing is planned.
- PID-based attestation cannot fully prevent PID-reuse attacks. For hardened deployments, hold process handles open for the session duration.

## Audit History

- **2026-03-27**: Full codebase audit (161+ findings) and security review (11 findings). All findings remediated. See `AUDIT_AND_FIXES_2026-03-27.md`.
