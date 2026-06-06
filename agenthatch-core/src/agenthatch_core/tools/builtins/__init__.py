"""Builtin tool registry (agenthatch-core).

Builtin tools are the "standard library" that all agents can require.
Each entry maps a tool name to a class that implements the tool.
"""

from __future__ import annotations

from typing import Any

# Registry of builtin tool classes, keyed by tool name.
# Agenthatch-core provides stubs; the full implementations live in agenthatch.
BUILTIN_REGISTRY: dict[str, Any] = {}


def register_builtin(name: str, cls: Any) -> None:
    """Register a builtin tool class."""
    BUILTIN_REGISTRY[name] = cls