"""Shared types and protocols (agenthatch-core)."""

from typing import Any, Protocol


class AgentIdentity:
    """Simple container for agent identity.

    Used by AHCoreAgent for independent agents.
    """
    def __init__(self, id: str, display_name: str, version: str = ""):
        self.id = id
        self.display_name = display_name
        self.version = version


class SpecProtocol(Protocol):
    """Protocol for anything that looks like an AHSSpec for ContextManager.

    Follows the AHSSPEC structure:
    - spec.identity: contains id, display_name, version
    - spec.intent: contains triggers, satisfies, summary
    - spec.instructions: contains workflow, rules, safety, output_template
    - spec.interface: contains provides, requires, mcp_servers
    - spec.resources: contains scripts, references, assets
    """
    identity: Any
    intent: Any
    instructions: Any
    interface: Any
    resources: Any
