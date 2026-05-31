"""agenthatch Agent Runtime — v0.5.

SkillAgent is the entry point. Loads from AHSSPEC, assembles
capabilities, and provides chat/chat_stream interface.

v0.5: + CompactSummary, HooksManager, HookPoint, StateManager, SessionState
"""

from agenthatch.agent.compact import CompactSummary
from agenthatch.agent.hooks import HookPoint, HooksManager
from agenthatch.agent.offload import SessionState, StateManager
from agenthatch.agent.runtime import SkillAgent

__all__ = [
    "SkillAgent",
    "CompactSummary",
    "HookPoint",
    "HooksManager",
    "SessionState",
    "StateManager",
]
