"""Production-style runtime host for AstraWeave IPC service loops.

The host wraps :class:`AstraWeaveIpcServer` with a small amount of lifecycle
management so tests and future CLI entrypoints can start, inspect, and stop the
runtime deterministically without depending on additional infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event, RLock
from time import monotonic
from typing import Any
from urllib.parse import urlparse
import os

from .ipc_server import AstraWeaveIpcServer, _is_loopback_host
from .service import AstraWeaveService


def _require_loopback_host(host: str) -> str:
    normalized = (host or "").strip() or "127.0.0.1"
    if not _is_loopback_host(normalized):
        raise ValueError("endpoint host must be localhost or loopback in v1")
    return normalized


@dataclass(frozen=True, slots=True)
class ServiceHostConfig:
    """Configuration for the AstraWeave runtime host."""

    endpoint: str | None = None
    prefer_named_pipe: bool = False
    pipe_name: str = r"\\.\pipe\astrawave"
    host: str = "127.0.0.1"
    port: int = 0
    authkey: bytes | None = None
    poll_interval_seconds: float = 0.05

    def __post_init__(self) -> None:
        if self.host is not None and not str(self.host).strip():
            raise ValueError("host must be a non-empty string when provided")
        if self.port < 0:
            raise ValueError("port must be zero or a positive integer")
        if not self.pipe_name or not self.pipe_name.strip():
            raise ValueError("pipe_name must be a non-empty string")
        if self.poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")


@dataclass(frozen=True, slots=True)
class ServiceHostStatus:
    """Snapshot of host runtime state for diagnostics and tests."""

    running: bool
    transport: str | None
    endpoint: Any
    uptime_seconds: float | None
    served_connections: int
    served_requests: int


class AstraWeaveServiceHost:
    """Manage the lifecycle of an AstraWeave IPC runtime host."""

    def __init__(
        self,
        config: ServiceHostConfig | None = None,
        *,
        service: AstraWeaveService | None = None,
        server: AstraWeaveIpcServer | None = None,
    ) -> None:
        self._config = config or ServiceHostConfig()
        self._service = service
        self._server = server
        self._lock = RLock()
        self._stop_event = Event()
        self._started_at: float | None = None
        self._stopped_at: float | None = None

    @property
    def config(self) -> ServiceHostConfig:
        """Return the host configuration."""

        return self._config

    @property
    def service(self) -> AstraWeaveService:
        """Return the wrapped service instance, creating one if needed."""

        with self._lock:
            if self._service is None and self._server is not None:
                self._service = self._server.service
            if self._service is None:
                self._service = AstraWeaveService()
            return self._service

    @property
    def server(self) -> AstraWeaveIpcServer | None:
        """Expose the current server instance, if one has been created."""

        return self._server

    @property
    def is_running(self) -> bool:
        """Return whether the host-managed server loop is active."""

        server = self._server
        return server is not None and server.is_running

    def __enter__(self) -> "AstraWeaveServiceHost":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def start(self) -> ServiceHostStatus:
        """Start the underlying IPC server if it is not already running."""

        with self._lock:
            server = self._ensure_server()
            if server.is_running:
                return self.status()

            self._stop_event.clear()
            self._stopped_at = None
            try:
                server.start()
            except Exception:
                self._stopped_at = monotonic()
                self._stop_event.set()
                raise
            self._started_at = monotonic()
            return self.status()

    def stop(self) -> ServiceHostStatus:
        """Stop the host and release the underlying server resources."""

        with self._lock:
            self._stop_event.set()
            server = self._server
            if server is not None:
                try:
                    server.stop()
                finally:
                    self._stopped_at = monotonic()
            else:
                self._stopped_at = monotonic()
            return self.status()

    def serve_forever(self) -> ServiceHostStatus:
        """Run until :meth:`stop` is called or the server exits unexpectedly."""

        self.start()
        while not self._stop_event.wait(self._config.poll_interval_seconds):
            server = self._server
            if server is None or not server.is_running:
                break
        return self.status()

    def run_for(self, duration_seconds: float) -> ServiceHostStatus:
        """Run the host for a bounded amount of time and then stop it."""

        if duration_seconds < 0:
            raise ValueError("duration_seconds must be zero or positive")
        self.start()
        self._stop_event.wait(duration_seconds)
        return self.stop()

    def status(self) -> ServiceHostStatus:
        """Return a structured status snapshot for diagnostics."""

        server = self._server
        running = server.is_running if server is not None else False
        uptime_seconds = self._compute_uptime_seconds(running=running)
        transport = server.transport if server is not None else None
        endpoint = server.endpoint if server is not None else self._planned_endpoint()
        served_connections = int(getattr(server, "_served_connections", 0)) if server is not None else 0
        served_requests = int(getattr(server, "_served_requests", 0)) if server is not None else 0
        return ServiceHostStatus(
            running=running,
            transport=transport,
            endpoint=endpoint,
            uptime_seconds=uptime_seconds,
            served_connections=served_connections,
            served_requests=served_requests,
        )

    @property
    def endpoint(self) -> Any:
        """Expose the current or planned endpoint."""

        return self.status().endpoint

    @property
    def transport(self) -> str | None:
        """Expose the current or planned transport."""

        return self.status().transport

    def _ensure_server(self) -> AstraWeaveIpcServer:
        with self._lock:
            if self._server is not None:
                if self._service is None:
                    self._service = self._server.service
                return self._server

            prefer_named_pipe, pipe_name, host, port = self._resolve_transport_settings()
            self._server = AstraWeaveIpcServer(
                service=self.service,
                prefer_named_pipe=prefer_named_pipe,
                pipe_name=pipe_name,
                host=host,
                port=port,
                authkey=self._config.authkey,
            )
            return self._server

    def _resolve_transport_settings(self) -> tuple[bool, str, str, int]:
        endpoint = (self._config.endpoint or "").strip()
        if not endpoint or endpoint == "auto":
            if os.name == "nt" and self._config.prefer_named_pipe:
                return True, self._config.pipe_name, self._config.host, self._config.port
            host = _require_loopback_host(self._config.host)
            return False, self._config.pipe_name, host, self._config.port

        if endpoint.startswith("pipe://"):
            return True, endpoint.removeprefix("pipe://"), self._config.host, self._config.port

        if endpoint.startswith("\\\\.\\pipe\\"):
            return True, endpoint, self._config.host, self._config.port

        parsed = urlparse(endpoint)
        if parsed.scheme == "tcp":
            host = _require_loopback_host(parsed.hostname or self._config.host)
            try:
                parsed_port = parsed.port
            except ValueError as exc:
                raise ValueError(f"invalid TCP endpoint port: {endpoint}") from exc
            port = parsed_port if parsed_port is not None else self._config.port or 0
            return False, self._config.pipe_name, host, port

        if ":" in endpoint and not endpoint.startswith("["):
            host, _, port_text = endpoint.rpartition(":")
            if not host:
                host = self._config.host
            host = _require_loopback_host(host)
            try:
                port = int(port_text)
            except ValueError as exc:
                raise ValueError(f"invalid TCP endpoint port: {endpoint}") from exc
            return False, self._config.pipe_name, host, port

        if os.name == "nt" and self._config.prefer_named_pipe:
            return True, endpoint, self._config.host, self._config.port

        return False, self._config.pipe_name, _require_loopback_host(endpoint), self._config.port

    def _planned_endpoint(self) -> Any:
        prefer_named_pipe, pipe_name, host, port = self._resolve_transport_settings()
        if prefer_named_pipe:
            return pipe_name
        return (host, port)

    def _compute_uptime_seconds(self, *, running: bool) -> float | None:
        if self._started_at is None:
            return None
        if running:
            return round(max(0.0, monotonic() - self._started_at), 6)
        if self._stopped_at is not None:
            return round(max(0.0, self._stopped_at - self._started_at), 6)
        return None


__all__ = [
    "AstraWeaveServiceHost",
    "ServiceHostConfig",
    "ServiceHostStatus",
]
