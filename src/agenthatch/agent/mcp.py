"""MCP (Model Context Protocol) client for SkillAgent external tool integration.

Each MCP server is a stdio or SSE process. The client discovers tools via
list_tools() and exposes them as CapBus external handlers.
"""

from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """MCP server configuration — mirrors Claude Code's mcp.json format."""
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    transport: str = "stdio"


@dataclass
class MCPToolDef:
    """MCP tool definition from list_tools() response."""
    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)


class MCPClient:
    """Per-skill MCP client that manages server processes and tool dispatch."""

    def __init__(self) -> None:
        self._servers: dict[str, MCPServerConfig] = {}
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._tools: dict[str, MCPToolDef] = {}

    def add_server(self, name: str, config: MCPServerConfig) -> None:
        self._servers[name] = config

    def connect_all(self) -> None:
        for sname, cfg in self._servers.items():
            try:
                proc = subprocess.Popen(
                    [cfg.command] + cfg.args,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env={**cfg.env},
                )
                self._processes[sname] = proc
                self._discover_tools(sname, proc)
            except Exception as e:
                logger.error("MCP server %s failed to start: %s", sname, e)

    def _discover_tools(
        self, sname: str, proc: subprocess.Popen[str]
    ) -> None:
        resp = self._send_request(proc, {"method": "tools/list", "params": {}})
        for tool in resp.get("tools", []):
            full_name = f"mcp__{sname}__{tool['name']}"
            self._tools[full_name] = MCPToolDef(
                name=tool["name"],
                description=tool.get("description", ""),
                input_schema=tool.get("inputSchema", {}),
            )

    def _send_request(
        self, proc: subprocess.Popen[str], request: dict[str, Any]
    ) -> dict[str, Any]:
        payload = json.dumps(request)
        assert proc.stdin is not None
        assert proc.stdout is not None
        proc.stdin.write(payload + "\n")
        proc.stdin.flush()
        while True:
            line = proc.stdout.readline()
            if not line:
                return {}
            try:
                return cast(dict[str, Any], json.loads(line))
            except json.JSONDecodeError:
                continue

    def call_tool(
        self, server_name: str, tool_name: str, arguments: dict[str, Any]
    ) -> str:
        proc = self._processes.get(server_name)
        if not proc:
            return f"Error: MCP server '{server_name}' not connected"
        resp = self._send_request(
            proc,
            {
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            },
        )
        if "error" in resp:
            return f"MCP error: {resp['error']}"
        return json.dumps(resp.get("content", resp.get("result", "")))

    def list_tool_definitions(self) -> list[dict[str, Any]]:
        """Return tool definitions in OpenAI function-calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": full_name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for full_name, t in self._tools.items()
        ]

    def register_with_capbus(self, capbus: Any) -> None:
        """Register all MCP tools as external handlers on CapBus."""
        for full_name, t in self._tools.items():
            sname = full_name.split("__")[1]
            tool_name = t.name

            def make_handler(sn: str, tn: str) -> Callable[..., str]:
                def handler(**kwargs: Any) -> str:
                    return self.call_tool(sn, tn, kwargs)
                return handler

            capbus.register_external_tool(
                full_name, t.input_schema, make_handler(sname, tool_name)
            )

    def disconnect_all(self) -> None:
        for _sname, proc in self._processes.items():
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        self._processes.clear()
