"""Capability Bus — v0.4 runtime capability registry and router.

Provides complete register/match/route/inject_builtin/list_tool_definitions.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
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
    _external_handlers: dict[str, Callable[..., str]] = field(default_factory=dict)
    _external_schemas: dict[str, dict[str, Any]] = field(default_factory=dict)

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
        else:
            logger.warning("inject_builtin: '%s' not found in BUILTIN_REGISTRY", name)

    def register_external_tool(
        self, name: str, schema: dict[str, Any], handler: Callable[..., str]
    ) -> None:
        """Register an external tool (MCP or API template) with a handler.

        The handler receives **kwargs and returns a string result.
        """
        self._external_handlers[name] = handler
        self._external_schemas[name] = schema

    def match(self, required: str) -> CapabilityRegistration | None:
        """Match a requirement to a registered capability."""
        if required in self.capabilities:
            return self.capabilities[required]
        return fuzzy_match(required, self.capabilities)

    def route(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute routing: tool_call → capability execution."""

        # v0.6: task_complete signal tool — LLM calls this to mark task done
        if tool_name == "task_complete":
            return "Task completed."

        # DD-05-11: External handlers first (most specific)
        if tool_name in self._external_handlers:
            return self._external_handlers[tool_name](**arguments)

        cap = self.capabilities.get(tool_name)
        if cap and cap.executor:
            executor: Any = cap.executor
            if hasattr(executor, "execute"):
                return str(executor.execute(**arguments))
            if hasattr(executor, "execute_script"):
                script_name = str(arguments.get("script_name", ""))
                filtered_args = {k: v for k, v in arguments.items() if k != "script_name"}
                return str(executor.execute_script(
                    script_name, **filtered_args
                ))

        builtin: Any = self.builtins.get(tool_name)
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
        """Return LLM-visible tool definitions.

        Only builtins and external handlers are exposed.
        Capability declarations (provides) are metadata-only.
        """
        definitions: list[dict[str, Any]] = []

        for name, cap in self.builtins.items():
            schema = getattr(cap, "schema", {})
            definitions.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": getattr(cap, "description", ""),
                    "parameters": schema,
                },
            })

        for name, schema in self._external_schemas.items():
            definitions.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"External tool: {name}",
                    "parameters": schema,
                },
            })

        # v0.6: task_complete signal tool — LLM must explicitly call this to terminate
        definitions.append({
            "type": "function",
            "function": {
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
                            "description": "Brief summary of what was accomplished.",
                        },
                    },
                    "required": ["summary"],
                },
            },
        })

        return definitions


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


class APITemplateExecutor:
    """Executes API templates via the http_client builtin."""

    def __init__(self, template: Any, http_client: Any):
        self._tpl = template
        self._http = http_client

    def build_url(self, **kwargs: Any) -> str:
        import string
        placeholders = [
            t[1] for t in string.Formatter().parse(str(self._tpl.url)) if t[1]
        ]
        filtered = {k: v for k, v in kwargs.items() if k in placeholders}
        try:
            return str(self._tpl.url).format(**filtered)
        except KeyError as e:
            logger.warning("build_url: missing placeholder %s in URL template", e)
            # Replace missing placeholders with empty string
            safe_url = str(self._tpl.url)
            for ph in placeholders:
                if ph not in filtered:
                    safe_url = safe_url.replace("{" + ph + "}", "")
            return safe_url.format(**filtered)

    def build_headers(self) -> dict[str, str]:
        headers = dict(self._tpl.headers)
        if self._tpl.auth_env_var:
            import os
            token = os.environ.get(self._tpl.auth_env_var, "")
            if token:
                headers["Authorization"] = f"Bearer {token}"
        return headers

    def execute(self, **kwargs: Any) -> str:
        url = self.build_url(**kwargs)
        headers = self.build_headers()
        return str(self._http.execute(
            method=self._tpl.method, url=url, headers=headers
        ))
