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
                return str(cap.executor.execute(**arguments))
            if hasattr(cap.executor, "execute_script"):
                script_name = arguments.get("script_name", "")
                filtered_args = {k: v for k, v in arguments.items() if k != "script_name"}
                return str(cap.executor.execute_script(
                    script_name, **filtered_args
                ))

        builtin = self.builtins.get(tool_name)
        if builtin:
            return str(builtin.execute(**arguments))

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
                properties: dict[str, Any] = {}
                for k, v in schema.items():
                    result = _json_type(v)
                    if isinstance(result, dict):
                        properties[k] = result
                    else:
                        properties[k] = {"type": result}
                schema = {
                    "type": "object",
                    "properties": properties,
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


def _json_type(value: Any) -> str | dict[str, Any]:
    """Convert Python type annotation to JSON Schema type.

    Handles arbitrary nesting: scalars, arrays, objects, nested arrays of objects.
    Returns a string for scalars, a dict for compound types (no json.dumps).
    """
    TYPE_MAP: dict[str, str] = {
        "string": "string", "integer": "integer", "number": "number",
        "boolean": "boolean", "object": "object", "array": "array",
    }

    if isinstance(value, list) and len(value) > 0:
        first = value[0]
        if isinstance(first, str):
            item_type = TYPE_MAP.get(first.lower(), "string")
            return {"type": "array", "items": {"type": item_type}}
        elif isinstance(first, dict):
            item_props: dict[str, Any] = {}
            for k, v in first.items():
                result = _json_type(v)
                if isinstance(result, dict):
                    item_props[k] = result
                else:
                    item_props[k] = {"type": result}
            return {
                "type": "array",
                "items": {"type": "object", "properties": item_props},
            }
        else:
            return {"type": "array", "items": {"type": "string"}}

    if isinstance(value, dict):
        if "items" in value and isinstance(value["items"], list):
            return {
                "type": "array",
                "items": {"type": _json_type(value["items"][0])},
            }
        obj_props: dict[str, Any] = {}
        for k, v in value.items():
            result = _json_type(v)
            if isinstance(result, dict):
                obj_props[k] = result
            else:
                obj_props[k] = {"type": result}
        return {"type": "object", "properties": obj_props}

    return TYPE_MAP.get(str(value).lower(), "string")
