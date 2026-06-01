"""agenthatch Agent Runtime — v0.5.6.

SkillAgent is the entry point. Loads from AHSSPEC, assembles
capabilities, and provides chat/chat_stream interface.

v0.5.6: + MCPClient, MCPServerConfig, MCPToolDef, Checkpoint, CheckpointManager,
APITemplateExecutor, external tool dispatch, retry, circuit breaker,
anchor rules, output template guard, batch progress gating.
"""

from agenthatch.agent.compact import CompactSummary
from agenthatch.agent.hooks import HookPoint, HooksManager
from agenthatch.agent.mcp import MCPClient, MCPServerConfig, MCPToolDef
from agenthatch.agent.offload import Checkpoint, CheckpointManager, SessionState, StateManager
from agenthatch.agent.runtime import SkillAgent
from agenthatch.cap.bus import APITemplateExecutor

__all__ = [
    "SkillAgent",
    "CompactSummary",
    "HookPoint",
    "HooksManager",
    "SessionState",
    "StateManager",
    "MCPClient",
    "MCPServerConfig",
    "MCPToolDef",
    "Checkpoint",
    "CheckpointManager",
    "APITemplateExecutor",
]
