"""agenthatch-core exception hierarchy."""

from __future__ import annotations


class CoreError(Exception):
    """Base agenthatch-core exception."""
    exit_code = 1


class ConfigError(CoreError):
    """Configuration file error."""
    exit_code = 2


class ProviderNotFoundError(CoreError):
    """Requested provider not found in registry."""
    exit_code = 2


class ApiKeyError(CoreError):
    """API key is missing or failed verification."""
    exit_code = 1


class ToolCallError(CoreError):
    """LLM tool call execution failed."""
    exit_code = 9


class ProviderCapabilityError(CoreError):
    """Provider does not support a required capability."""
    exit_code = 10


class CapabilityNotFoundError(CoreError):
    """Requested capability not registered on the CapBus."""
    exit_code = 11


class CompactError(CoreError):
    """Auto-compact operation failed."""
    exit_code = 13


class AgentRuntimeError(CoreError):
    """Agent runtime error."""
    exit_code = 7