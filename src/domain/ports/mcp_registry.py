"""Outbound port for the mcp-raganything MCP-server registry (REQ-017).

The registry remains the sole owner of MCP server persistence, validation,
generation, mounting and rehydration. This port lets ``composable-agents``
call the owning application through a trusted HTTP adapter using the ORIGINATING
credential — never a service account (REQ-025).

Two families of operations with deliberately different secrecy guarantees:

* Management (list/get/create/update/delete/validate) — the adapter forwards
  the originating credential to the configured registry base URL and returns
  the registry's MASKED projections (REQ-018). Secrets in write inputs
  (REQ-019) are write-only.
* Execution-config resolution (:meth:`get_execution_config`) — a TRUSTED
  server-side operation (REQ-037): the adapter may use privileged registry
  internals (the authenticated reveal endpoint) to resolve decrypted
  connection details used ONLY to load tools inside the runner. Those values
  never appear in management responses, traces or logs.
"""

import asyncio
from abc import ABC, abstractmethod

from src.domain.entities.mcp_server_config import McpServerConfig
from src.domain.entities.registered_mcp_server import (
    McpServerValidationReport,
    McpServerWrite,
    RegisteredMcpServer,
)


class McpRegistryClient(ABC):
    """Abstract client for the trusted mcp-raganything registry REST API."""

    @abstractmethod
    async def list_servers(self) -> list[RegisteredMcpServer]:
        """List the current user's registered MCP servers (masked)."""
        ...

    @abstractmethod
    async def get_server(self, name: str) -> RegisteredMcpServer:
        """Return one registered MCP server (masked).

        Raises:
            McpRegistryNotFoundError: Missing or foreign server.
        """
        ...

    @abstractmethod
    async def create_server(self, payload: McpServerWrite) -> RegisteredMcpServer:
        """Register an external MCP server or generate one from OpenAPI/Swagger."""
        ...

    @abstractmethod
    async def update_server(self, name: str, changes: dict) -> RegisteredMcpServer:
        """Update a registered MCP server (masked response)."""
        ...

    @abstractmethod
    async def delete_server(self, name: str) -> None:
        """Delete a registered MCP server."""
        ...

    @abstractmethod
    async def validate_server(self, payload: McpServerWrite) -> McpServerValidationReport:
        """Validate a candidate server without persisting it."""
        ...

    @abstractmethod
    async def get_execution_config(self, name: str) -> McpServerConfig:
        """Resolve the (decrypted) execution configuration of a registered server.

        TRUSTED path (REQ-037): used only for server-side tool loading inside
        the runner. Decrypted headers/env/tokens are never exposed through
        management responses, traces or logs.

        Raises:
            McpRegistryNotFoundError: Missing or foreign server.
        """
        ...

    @abstractmethod
    async def reveal_server(self, name: str) -> dict:
        """Resolve the RAW (decrypted) registry entry of a registered server.

        TRUSTED path (REQ-037 family): returns the reveal payload WITHOUT
        placeholder resolution, so caller-side checks (e.g. the cron
        unattended-safety scan) can pattern-match service-side placeholders
        exactly as stored. The payload is consumed server-side only and its
        secret material is never logged.

        Raises:
            McpRegistryNotFoundError: Missing or foreign server.
        """
        ...

    async def aclose(self) -> None:  # noqa: B027 — optional lifecycle hook
        """Release any owned HTTP resources (composition-root lifecycle hook).

        Concrete adapters with a persistent connection pool override this;
        stateless clients keep the default cooperative no-op.
        """
        await asyncio.sleep(0)
