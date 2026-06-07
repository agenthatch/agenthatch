"""Capability Bus — runtime capability registry and router (agenthatch-core)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agenthatch_core.exceptions import CapabilityNotFoundError
from agenthatch_core.tools.marshal import fuzzy_match

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
class ToolDefinition:
    """OpenAI-compatible tool definition."""
    type: str = "function"
    function: dict[str, Any] = field(default_factory=dict)


@dataclass
class CapBus:
    """Capability Bus — runtime tool registry and router."""

    capabilities: dict[str, CapabilityRegistration] = field(default_factory=dict)
    builtins: dict[str, Any] = field(default_factory=dict)
    unavailable: set[str] = field(default_factory=set)
    _external_handlers: dict[str, Callable[..., str]] = field(default_factory=dict)
    _external_schemas: dict[str, dict[str, Any]] = field(default_factory=dict)

    def register(
        self,
        name: str,
        executor: Callable[..., str] | None = None,
        schema: dict[str, Any] | None = None,
        source: str | None = None,
        cap_type: str = "",
    ) -> None:
        """Register a capability on the bus."""
        self.capabilities[name] = CapabilityRegistration(
            name=name,
            type=cap_type,
            schema=schema or {},
            source_skill=source or "",
            executor=executor,
        )

    def register_external_tool(
        self, name: str, schema: dict[str, Any], handler: Callable[..., str]
    ) -> None:
        """Register an external tool handler."""
        self._external_handlers[name] = handler
        self._external_schemas[name] = schema

    def inject_builtin(self, name: str, instance: Any) -> None:
        """Inject a builtin tool instance."""
        self.builtins[name] = instance

    def route(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Route a tool call to the appropriate executor."""

        # v0.6: task_complete signal tool — LLM calls this to mark task done
        if tool_name == "task_complete":
            return "Task completed."

        # 1. Check capabilities
        cap = self.capabilities.get(tool_name)
        if cap is None:
            cap = fuzzy_match(tool_name, self.capabilities)

        if cap is not None and cap.executor is not None:
            try:
                return str(cap.executor(arguments))
            except Exception as e:
                logger.warning("Tool '%s' execution failed: %s", tool_name, e)
                return f"Error executing tool '{tool_name}': {e}"

        # 2. Check builtins
        if tool_name in self.builtins:
            builtin = self.builtins[tool_name]
            if hasattr(builtin, "execute"):
                try:
                    return str(builtin.execute(**arguments))
                except Exception as e:
                    return f"Error executing builtin '{tool_name}': {e}"

        # 3. Check external handlers
        if tool_name in self._external_handlers:
            try:
                return str(self._external_handlers[tool_name](**arguments))
            except Exception as e:
                return f"Error executing external tool '{tool_name}': {e}"

        # 4. Check unavailable
        if tool_name in self.unavailable:
            return f"Tool '{tool_name}' is not available in this agent."

        raise CapabilityNotFoundError(f"Tool '{tool_name}' not found on CapBus")

    def list_tool_definitions(self) -> list[ToolDefinition]:
        """List all tool definitions for LLM function calling."""
        tools: list[ToolDefinition] = []

        for name, cap in self.capabilities.items():
            if name in self.unavailable:
                continue
            params = cap.schema.get("parameters", cap.schema)
            tools.append(ToolDefinition(
                function={
                    "name": name,
                    "description": cap.schema.get("description", name),
                    "parameters": _normalize_json_schema(params),
                }
            ))

        for name, schema in self._external_schemas.items():
            params = schema.get("parameters", schema)
            tools.append(ToolDefinition(
                function={
                    "name": name,
                    "description": schema.get("description", name),
                    "parameters": _normalize_json_schema(params),
                }
            ))

        for name, builtin in self.builtins.items():
            if hasattr(builtin, "tool_definition"):
                tools.append(builtin.tool_definition)
            elif hasattr(builtin, "schema"):
                tools.append(ToolDefinition(
                    function={
                        "name": name,
                        "description": getattr(builtin, "description", name),
                        "parameters": _normalize_json_schema(
                            getattr(builtin, "schema", {})
                        ),
                    }
                ))

        # v0.6: task_complete signal tool — LLM must explicitly call this to terminate
        tools.append(ToolDefinition(
            function={
                "name": "task_complete",
                "description": (
                    "Call this tool when the user's request has been fully completed. "
                    "Only invoke this after all required steps are done and results are confirmed."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Brief summary of what was accomplished (1-2 sentences).",
                        },
                    },
                    "required": ["summary"],
                },
            }
        ))

        return tools

    def mark_unavailable(self, name: str) -> None:
        """Mark a tool as unavailable."""
        self.unavailable.add(name)


def _normalize_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Ensure the schema is a valid JSON Schema object with type: object wrapper.

    DeepSeek and other strict providers require parameters to have
    ``type: "object"`` at the root level.
    """
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}

    # Already has type field at root level
    if "type" in schema:
        return schema

    # Wrap flat key-type pairs into proper JSON Schema
    if "properties" not in schema:
        # Check if schema looks like flat params: {"city": {"type": "string"}}
        # or already has properties
        has_nested = any(
            isinstance(v, dict) and "type" in v
            for v in schema.values()
        )
        if has_nested:
            return {
                "type": "object",
                "properties": schema,
                "required": list(schema.keys()),
            }

    return {
        "type": "object",
        "properties": schema.get("properties", {}),
        "required": schema.get("required", []),
    }