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
    _guard: Any = None  # v0.7.6: CompiledGuard for pre-tool validation
    _output_schemas: dict[str, dict[str, Any]] = field(default_factory=dict)  # v0.7.6

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

        # v0.7.6: Pre-tool validation via CompiledGuard
        if self._guard is not None:
            allowed, msg = self._guard.check_pre_tool_call(tool_name, arguments)
            if not allowed:
                logger.warning("Pre-tool validation blocked: %s -> %s", tool_name, msg)
                return f"Error: {msg}"

        result: str | None = None

        # 1. Check capabilities
        cap = self.capabilities.get(tool_name)
        if cap is None:
            cap = fuzzy_match(tool_name, self.capabilities)

        if cap is not None and cap.executor is not None:
            try:
                result = str(cap.executor(arguments))
            except Exception as e:
                logger.warning("Tool '%s' execution failed: %s", tool_name, e)
                return f"Error executing tool '{tool_name}': {e}"

        # 2. Check builtins
        if result is None and tool_name in self.builtins:
            builtin = self.builtins[tool_name]
            if hasattr(builtin, "execute"):
                try:
                    result = str(builtin.execute(**arguments))
                except Exception as e:
                    return f"Error executing builtin '{tool_name}': {e}"

        # 3. Check external handlers
        if result is None and tool_name in self._external_handlers:
            try:
                result = str(self._external_handlers[tool_name](**arguments))
            except Exception as e:
                return f"Error executing external tool '{tool_name}': {e}"

        # 4. Check unavailable
        if result is None and tool_name in self.unavailable:
            return f"Tool '{tool_name}' is not available in this agent."

        if result is None:
            # v0.7.11: Distinguish "no executor" vs "truly not found"
            if cap is not None:
                # Capability exists but has no executor (e.g., mcp_proxy not connected)
                raise CapabilityNotFoundError(
                    f"Tool '{tool_name}' is registered but has no executor. "
                    f"The capability backend may not be available or connected."
                )
            raise CapabilityNotFoundError(f"Tool '{tool_name}' not found on CapBus")

        # v0.7.6: Validate tool output against output_schema
        result = self._validate_output(tool_name, result)

        return result

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

    def _validate_output(self, tool_name: str, result: str) -> str:
        """v0.7.6: Validate tool output against output_schema if configured.

        Checks that the result is valid JSON and field types match the
        declared schema. Catches formatting errors before they confuse the LLM.

        If output_schema.type is "string", the result is returned as-is
        without JSON parsing (plain text output).
        """
        import json

        output_schema = self._output_schemas.get(tool_name)
        if output_schema is None:
            return result

        # If schema declares type: string, output is plain text — no JSON parsing
        if isinstance(output_schema, dict) and output_schema.get("type") == "string":
            return result

        try:
            data = json.loads(result)
        except json.JSONDecodeError as e:
            trimmed = result.strip()
            if not trimmed:
                return (
                    f"MCP server returned empty response for '{tool_name}'. "
                    "Check that mcporter is running and the MCP server is configured."
                )
            logger.debug("Tool %s output JSON decode failed: %s", tool_name, e)
            return f"Tool '{tool_name}' returned non-JSON: {trimmed[:100]}"

        if isinstance(data, dict) and "properties" in output_schema:
            _JSON_TYPE_MAP = {
                "str": "string", "int": "integer", "float": "number",
                "bool": "boolean", "list": "array", "dict": "object",
            }
            for key, spec in output_schema["properties"].items():
                expected = spec.get("type", "string")
                if key in data:
                    actual = _JSON_TYPE_MAP.get(type(data[key]).__name__, "string")
                    if actual != expected:
                        logger.warning(
                            "Tool %s output schema validation failed: field '%s' expected %s, got %s",
                            tool_name, key, expected, actual,
                        )
                        return (
                            f"Error: Tool '{tool_name}' output field '{key}' "
                            f"expected {expected}, got {actual}"
                        )

        return json.dumps(data, indent=2)


def _normalize_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Ensure the schema is a valid JSON Schema object with type: object wrapper.

    DeepSeek and other strict providers require parameters to have
    ``type: "object"`` at the root level.
    """
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}

    # Already a valid object schema at root
    if schema.get("type") == "object":
        return schema

    # Has a non-object type at root (number, string, array, etc.) — wrap it
    if "type" in schema and schema["type"] != "object":
        return {
            "type": "object",
            "properties": {"value": schema},
            "required": ["value"],
        }

    # No type field at root — wrap flat properties if they look like params
    if "properties" not in schema:
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
        # Handle flat key-value objects where values are type strings
        # e.g. {"doc_id": "string", "query": "string"} →
        #      {"type": "object", "properties": {"doc_id": {"type": "string"}, ...}, "required": [...]}
        if schema and all(isinstance(v, str) for v in schema.values()):
            return {
                "type": "object",
                "properties": {
                    k: {"type": v} for k, v in schema.items()
                },
                "required": list(schema.keys()),
            }
        # Handle flat objects where values are arrays (e.g. {"daily": [{}]})
        if schema and all(isinstance(v, list) for v in schema.values()):
            return {
                "type": "object",
                "properties": {
                    k: {"type": "array", "items": v[0] if v else {"type": "string"}}
                    for k, v in schema.items()
                },
                "required": list(schema.keys()),
            }

    return {
        "type": "object",
        "properties": schema.get("properties", {}),
        "required": schema.get("required", []),
    }