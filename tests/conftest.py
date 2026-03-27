"""Shared test helpers and fixtures for the AstraWeave test suite.

Addresses H23/H24: extracts duplicated URI-builder, server-polling, and
caller-identity helpers that were copy-pasted across multiple test files.
"""

from __future__ import annotations

import os
import time

from astrawave.security import CallerIdentity, resolve_process_user_sid


def endpoint_to_uri(endpoint: object) -> str:
    """Convert a server endpoint (tuple or string) to a URI string.

    Handles ``(host, port)`` tuples, named-pipe paths, and plain strings.
    """
    if isinstance(endpoint, tuple) and len(endpoint) == 2:
        host, port = endpoint
        return f"tcp://{host}:{port}"
    if isinstance(endpoint, str):
        if endpoint.startswith("\\\\.\\pipe\\"):
            return f"pipe://{endpoint}"
        if endpoint.startswith(("tcp://", "pipe://")):
            return endpoint
        return f"tcp://{endpoint}"
    raise AssertionError(f"Unsupported endpoint shape: {endpoint!r}")


def wait_for_server(server: object, retries: int = 40, delay: float = 0.05) -> str:
    """Poll *server.endpoint* until it becomes available and return its URI."""
    last_endpoint: object | None = None
    for _ in range(retries):
        last_endpoint = server.endpoint  # type: ignore[union-attr]
        if last_endpoint:
            return endpoint_to_uri(last_endpoint)
        time.sleep(delay)
    raise AssertionError(f"server did not expose an endpoint; last value={last_endpoint!r}")


def default_caller_identity() -> CallerIdentity:
    """Build a :class:`CallerIdentity` for the current process."""
    current_pid = os.getpid()
    current_sid = resolve_process_user_sid(current_pid) or "S-1-5-21-1000"
    return CallerIdentity(user_sid=current_sid, pid=current_pid)
