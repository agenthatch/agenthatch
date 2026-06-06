"""MCP (Model Context Protocol) tool loader (agenthatch-core).

Provides the ability to connect to MCP servers and register their tools
on the CapBus.  This is a stub; full implementation lives in agenthatch.
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
            - config: server-specific config (url, command, etc.)
    """
    logger.warning(
        "MCP loader not fully implemented in agenthatch-core. "
        "Server '%s' tools will not be available.",
        mcp_cfg.get("name", "unknown"),
    )