"""Context module."""

from agenthatch_core.context.compact import COMPACT_SYSTEM_PROMPT, CompactSummary
from agenthatch_core.context.manager import ContextManager

__all__ = ["ContextManager", "CompactSummary", "COMPACT_SYSTEM_PROMPT"]