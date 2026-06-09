"""MCP (Model Context Protocol) module for agenthatch-core.

Provides multi-transport MCP client, server configuration, and tool definitions.
"""

from agenthatch_core.mcp.client import (
    MCPClient,
    MCPToolDef,
    SSETransport,
    StdioTransport,
    StreamableHTTPTransport,
    Transport,
    _resolve_transport,
)
from agenthatch_core.mcp.config import MCPServerConfig

__all__ = [
    "MCPClient",
    "MCPServerConfig",
    "MCPToolDef",
    "Transport",
    "StdioTransport",
    "StreamableHTTPTransport",
    "SSETransport",
    "_resolve_transport",
]