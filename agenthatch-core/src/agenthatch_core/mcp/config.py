"""MCP (Model Context Protocol) configuration types for agenthatch-core."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MCPServerConfig:
    """MCP server configuration — transport-agnostic unified schema.

    Transports:
      - stdio: {command, args, env}
      - streamable_http: {url, headers, auth_token}
      - sse: {url, headers, auth_token}
    """
    transport: str = "stdio"
    timeout: float = 30.0
    # stdio fields
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    # HTTP/SSE fields
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    auth_token: str = ""