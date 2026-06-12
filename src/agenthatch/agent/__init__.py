"""agenthatch Agent Runtime — v0.5.7.

SkillAgent is the entry point. Loads from AHSSPEC, assembles
capabilities, and provides chat/chat_stream interface.

v0.5.7: + Multi-transport MCP (Stdio/StreamableHTTP/SSE), transport auto-detection,
graceful degradation, with_enriched_errors, reasoning_content probe,
provider auto-probing, brace-balanced JSON extraction, API template naming,
FLAT_CATALOG cleanup, Capability/Tool distinction, SkillBrick resource loading,
rich prompt mode.
"""

from agenthatch_core.hooks import HookPoint, HooksManager
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

from agenthatch.agent.compact import CompactSummary
from agenthatch.agent.offload import Checkpoint, CheckpointManager, StateManager
from agenthatch.agent.runtime import SkillAgent
from agenthatch.cap.bus import APITemplateExecutor

__all__ = [
    "SkillAgent",
    "CompactSummary",
    "HookPoint",
    "HooksManager",
    "StateManager",
    "MCPClient",
    "MCPServerConfig",
    "MCPToolDef",
    "Checkpoint",
    "CheckpointManager",
    "APITemplateExecutor",
    "Transport",
    "StdioTransport",
    "StreamableHTTPTransport",
    "SSETransport",
    "_resolve_transport",
]
