"""Capability Bus — v0.4 runtime capability registry and router.

Provides complete register/match/route/inject_builtin/list_tool_definitions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from agenthatch.cap.marshal import fuzzy_match
from agenthatch.exceptions import CapabilityNotFoundError

logger = logging.getLogger(__name__)


@dataclass
class CapabilityRegistration:
    """A capability registered on the bus."""
    name: str
    type: str
    schema: dict[str, Any] = field(default_factory=dict)
    source_skill: str = ""
    executor: Any = None


@dataclass
class CapBus:
    """Capability Bus — v0.4 real implementation."""

    capabilities: dict[str, CapabilityRegistration] = field(default_factory=dict)
    builtins: dict[str, Any] = field(default_factory=dict)
    unavailable: set[str] = field(default_factory=set)

    def register(
        self,
        name: str,
        cap_type: str,
        schema: dict[str, Any] | None = None,
        source_skill: str = "",
        executor: Any = None,
    ) -> None:
        """Register a capability on the bus."""
        self.capabilities[name] = CapabilityRegistration(
            name=name,
            type=cap_type,
            schema=schema or {},
            source_skill=source_skill,
            executor=executor,
        )

    def inject_builtin(self, name: str) -> None:
        """Inject a builtin capability."""
        from agenthatch.agent.builtins import BUILTIN_REGISTRY
        if name in BUILTIN_REGISTRY:
            self.builtins[name] = BUILTIN_REGISTRY[name]()

    def match(self, required: str) -> CapabilityRegistration | None:
        """Match a requirement to a registered capability."""
        if required in self.capabilities:
            return self.capabilities[required]
        return fuzzy_match(required, self.capabilities)

    def route(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute routing: tool_call → capability execution."""

        cap = self.capabilities.get(tool_name)
        if cap and cap.executor:
            if hasattr(cap.executor, "execute"):
                return cap.executor.execute(**arguments)
            if hasattr(cap.executor, "execute_script"):
                script_name = arguments.get("script_name", "")
                filtered_args = {k: v for k, v in arguments.items() if k != "script_name"}
                return cap.executor.execute_script(
                    script_name, **filtered_args
                )

        builtin = self.builtins.get(tool_name)
        if builtin:
            return builtin.execute(**arguments)

        raise CapabilityNotFoundError(
            f"Capability '{tool_name}' is not registered. "
            f"Available: {list(self.capabilities.keys()) + list(self.builtins.keys())}"
        )

    def mark_unavailable(self, name: str) -> None:
        """Mark a required capability as unavailable."""
        self.unavailable.add(name)

    def list_tool_definitions(self) -> list[dict[str, Any]]:
        """Generate OpenAI function calling tool definitions."""
        tools: list[dict[str, Any]] = []
        for cap in self.capabilities.values():
            schema = cap.schema
            if not schema.get("type"):
                schema = {
                    "type": "object",
                    "properties": {
                        k: {"type": _json_type(v)} for k, v in schema.items()
                    },
                    "required": list(schema.keys()),
                }
            tools.append({
                "type": "function",
                "function": {
                    "name": cap.name,
                    "description": f"[{cap.type}] from {cap.source_skill}",
                    "parameters": schema,
                },
            })
        for name, builtin in self.builtins.items():
            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"[builtin] {builtin.__class__.__name__}",
                    "parameters": builtin.schema,
                },
            })
        return tools


def _json_type(python_type_str: str) -> str:
    """Map AHSSPEC type shorthand to JSON Schema type."""
    mapping = {
        "string": "string",
        "number": "number",
        "integer": "integer",
        "boolean": "boolean",
        "array": "array",
        "object": "object",
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "list": "array",
        "dict": "object",
    }
    result = mapping.get(python_type_str)
    if result is None:
        logger.debug("Unknown type '%s', falling back to 'string'", python_type_str)
        return "string"
    return result
