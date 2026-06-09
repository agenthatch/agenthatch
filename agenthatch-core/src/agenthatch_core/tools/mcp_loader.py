"""MCP (Model Context Protocol) tool loader (agenthatch-core).

Connects to MCP servers and registers their tools on the CapBus.
Uses the agenthatch-core MCP client implementation.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def load_mcp_tools(capbus: Any, mcp_cfg: dict) -> None:
    """Connect to an MCP server and register its tools on the CapBus.

    Args:
        capbus: CapBus instance to register tools on.
        mcp_cfg: MCP server configuration dict with keys:
            - name: server name
            - config: server-specific config dict with keys:
                - command: command to run (for stdio transport)
                - args: list of arguments
                - env: environment variables dict
                - transport: "stdio", "streamable_http", or "sse"
                - url: server URL (for HTTP/SSE transports)
    """
    server_name = mcp_cfg.get("name", "unknown")
    server_config = mcp_cfg.get("config", {})

    try:
        from agenthatch_core.mcp.client import MCPClient
        from agenthatch_core.mcp.config import MCPServerConfig
    except ImportError:
        logger.warning(
            "MCP client not available in standalone mode. "
            "Server '%s' tools will not be available.",
            server_name,
        )
        return

    try:
        client = MCPClient()
        config = MCPServerConfig(
            command=server_config.get("command", ""),
            args=server_config.get("args", []),
            env=server_config.get("env", {}),
            transport=server_config.get("transport", "stdio"),
            url=server_config.get("url", ""),
            headers=server_config.get("headers", {}),
            auth_token=server_config.get("auth_token", ""),
            timeout=server_config.get("timeout", 30.0),
        )
        client.add_server(server_name, config)
        client.connect_all()
        client.register_with_capbus(capbus)
        logger.info("MCP server '%s' connected and tools registered.", server_name)
    except Exception as e:
        logger.warning(
            "Failed to connect MCP server '%s': %s",
            server_name, e,
        )
