"""agenthatch-core: Lightweight AI Agent runtime base.

Provides the universal chassis for agenthatch-generated Agents:
- LLM Hub (LLMClient, providers)
- Tool Bus (CapBus, tool registration/routing)
- Sandbox (subprocess execution)
- Conversation Loop (LLM <-> Tool cycle)
- Context Manager (system prompt, history, compaction)
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("agenthatch-core")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

from agenthatch_core.agent import AHCoreAgent
from agenthatch_core.llm.client import LLMClient
from agenthatch_core.llm.types import StreamDelta, ToolCallResponse
from agenthatch_core.tools.bus import CapBus, CapabilityRegistration, ToolDefinition
from agenthatch_core.sandbox.executor import Sandbox, SandboxConfig
from agenthatch_core.loop.agent_loop import ConversationLoop, RichToolCallEvent
from agenthatch_core.context.manager import ContextManager
from agenthatch_core.context.compact import CompactSummary

__all__ = [
    "__version__",
    "AHCoreAgent",
    "LLMClient",
    "StreamDelta",
    "ToolCallResponse",
    "CapBus",
    "CapabilityRegistration",
    "ToolDefinition",
    "Sandbox",
    "SandboxConfig",
    "ConversationLoop",
    "RichToolCallEvent",
    "ContextManager",
    "CompactSummary",
]