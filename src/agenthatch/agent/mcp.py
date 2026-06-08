"""MCP (Model Context Protocol) client for SkillAgent external tool integration.

v0.5.7: Multi-transport support (stdio, streamable_http, sse).
Transport abstraction layer with auto-detection and graceful degradation.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 30


# ── Transport ABC ────────────────────────────────────────────────────────


class Transport(ABC):
    """Abstract MCP transport — stdio, streamable_http, or sse."""

    @abstractmethod
    def connect(self) -> None:
        """Establish connection. Raise ConnectionError on failure."""
        ...

    @abstractmethod
    def send_request(self, request: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        """Send JSON-RPC request, return response dict.

        Returns {} on transport-level failure (connection lost).
        """
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection, release resources."""
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        """Return True if transport is currently connected."""
        ...


# ── MCPServerConfig ──────────────────────────────────────────────────────


@dataclass
class MCPServerConfig:
    """MCP server configuration — transport-agnostic unified schema.

    Transports:
      - stdio: {command, args, env}
      - streamable_http: {url, headers, auth_token}
      - sse: {url, headers, auth_token}
    """
    transport: str = "stdio"
    timeout: float = 30.0
    # stdio fields
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    # HTTP/SSE fields
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    auth_token: str = ""


# ── StdioTransport ───────────────────────────────────────────────────────


class StdioTransport(Transport):
    """Stdio MCP transport — launches server as subprocess."""

    def __init__(self, config: MCPServerConfig) -> None:
        self._config = config
        self._proc: subprocess.Popen[str] | None = None

    def connect(self) -> None:
        self._proc = subprocess.Popen(
            [self._config.command] + self._config.args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, **self._config.env},
        )

    def send_request(self, request: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        if not self._proc or not self._proc.stdin or not self._proc.stdout:
            return {}
        payload = json.dumps(request)
        self._proc.stdin.write(payload + "\n")
        self._proc.stdin.flush()
        effective_timeout = timeout if timeout is not None else self._config.timeout
        deadline = time.time() + effective_timeout
        while time.time() < deadline:
            line = self._proc.stdout.readline()
            if not line:
                return {}
            try:
                return cast(dict[str, Any], json.loads(line))
            except json.JSONDecodeError:
                continue
        logger.warning("MCP StdioTransport timed out after %.1fs", effective_timeout)
        return {}

    def disconnect(self) -> None:
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
            self._proc = None

    def is_connected(self) -> bool:
        return self._proc is not None and self._proc.poll() is None


# ── StreamableHTTPTransport ──────────────────────────────────────────────


class StreamableHTTPTransport(Transport):
    """Streamable HTTP MCP transport (MCP spec 2025-03-26).

    Single HTTP endpoint:
      - POST: send JSON-RPC request, receive JSON or SSE stream.
      - GET: establish SSE stream for server→client notifications.
    """

    def __init__(self, config: MCPServerConfig) -> None:
        self._config = config
        self._url = config.url.rstrip("/")
        self._headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if config.auth_token:
            self._headers["Authorization"] = f"Bearer {config.auth_token}"
        for k, v in config.headers.items():
            self._headers[k] = v
        self._client: Any = None
        self._session_id: str | None = None

    def connect(self) -> None:
        import httpx
        self._client = httpx.Client(timeout=self._config.timeout)
        init_resp = self.send_request({
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "agenthatch", "version": "0.5.7"},
            },
        })
        self._session_id = init_resp.get("result", {}).get("sessionId")
        if self._session_id:
            self._client.post(
                self._url,
                json={"jsonrpc": "2.0", "method": "notifications/initialized"},
                headers=self._headers,
            )

    def send_request(self, request: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        if not self._client:
            return {}
        if "jsonrpc" not in request:
            request = {"jsonrpc": "2.0", "id": 1, **request}
        try:
            resp = self._client.post(
                self._url, json=request, headers=self._headers,
                timeout=timeout if timeout is not None else self._config.timeout,
            )
            if resp.status_code == 200:
                content_type = resp.headers.get("content-type", "")
                if "text/event-stream" in content_type:
                    return self._parse_sse(resp.text)
                return cast(dict[str, Any], resp.json())
            logger.warning(
                "MCP HTTP %s returned %d: %s",
                self._url, resp.status_code, resp.text[:200],
            )
            return {}
        except Exception as e:
            logger.error("MCP HTTP request failed: %s", e)
            return {}

    def _parse_sse(self, text: str) -> dict[str, Any]:
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("data:"):
                data = line[5:].strip()
                try:
                    return cast(dict[str, Any], json.loads(data))
                except json.JSONDecodeError:
                    continue
        return {}

    def disconnect(self) -> None:
        if self._client:
            self._client.close()
            self._client = None
        self._session_id = None

    def is_connected(self) -> bool:
        return self._client is not None


# ── SSETransport ─────────────────────────────────────────────────────────


class SSETransport(Transport):
    """Legacy HTTP+SSE MCP transport (pre-2025-03-26).

    Uses two endpoints:
      - SSE endpoint: GET for server→client events
      - POST endpoint: send JSON-RPC requests
    """

    def __init__(self, config: MCPServerConfig) -> None:
        self._config = config
        self._url = config.url.rstrip("/")
        self._headers: dict[str, str] = {}
        if config.auth_token:
            self._headers["Authorization"] = f"Bearer {config.auth_token}"
        for k, v in config.headers.items():
            self._headers[k] = v
        self._client: Any = None

    def connect(self) -> None:
        import httpx
        self._client = httpx.Client(timeout=self._config.timeout)
        # MCP protocol handshake: initialize -> notifications/initialized
        init_response = self.send_request({
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "agenthatch", "version": "0.5.10"},
            },
        })
        if init_response:
            self.send_request({"method": "notifications/initialized"})

    def send_request(self, request: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        if not self._client:
            return {}
        if "jsonrpc" not in request:
            request = {"jsonrpc": "2.0", "id": 1, **request}
        try:
            resp = self._client.post(
                self._url, json=request, headers=self._headers,
                timeout=timeout if timeout is not None else self._config.timeout,
            )
            if resp.status_code == 200:
                return cast(dict[str, Any], resp.json())
            return {}
        except Exception as e:
            logger.error("MCP SSE request failed: %s", e)
            return {}

    def disconnect(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    def is_connected(self) -> bool:
        return self._client is not None


# ── Transport Registry ───────────────────────────────────────────────────


_TRANSPORT_ALIASES: dict[str, str] = {
    "http": "streamable_http",
    "https": "streamable_http",
    "streamable": "streamable_http",
}


_TRANSPORT_REGISTRY: dict[str, type[Transport]] = {
    "stdio": StdioTransport,
    "streamable_http": StreamableHTTPTransport,
    "sse": SSETransport,
}


def _resolve_transport(config: MCPServerConfig) -> str:
    """Auto-detect transport type from config fields, normalizing aliases."""
    if config.transport and config.transport != "auto":
        raw = config.transport.strip().lower()
        return _TRANSPORT_ALIASES.get(raw, raw)
    if config.url and not config.command:
        return _probe_transport(config)
    return "stdio"


def _probe_transport(config: MCPServerConfig) -> str:
    """Probe HTTP endpoint to distinguish streamable_http vs sse."""
    try:
        import httpx
        probe = httpx.post(
            config.url.rstrip("/"),
            json={
                "jsonrpc": "2.0", "id": 0, "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "agenthatch-probe", "version": "0.5.8"},
                },
            },
            headers={"Content-Type": "application/json"},
            timeout=5.0,
        )
        if probe.status_code == 200:
            data = probe.json()
            if "result" in data or "error" in data:
                return "streamable_http"
    except Exception:
        pass
    return "sse"


# ── MCPToolDef ───────────────────────────────────────────────────────────


@dataclass
class MCPToolDef:
    """MCP tool definition from list_tools() response."""
    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)


# ── MCPClient ────────────────────────────────────────────────────────────


class MCPClient:
    """Per-skill MCP client — multi-transport, lazy-connect, graceful degradation."""

    def __init__(self) -> None:
        self._servers: dict[str, MCPServerConfig] = {}
        self._transports: dict[str, Transport] = {}
        self._tools: dict[str, MCPToolDef] = {}
        self._unavailable: set[str] = set()

    def add_server(self, name: str, config: MCPServerConfig) -> None:
        self._servers[name] = config

    def connect_all(self) -> None:
        for sname, cfg in self._servers.items():
            transport_type = _resolve_transport(cfg)
            transport_cls = _TRANSPORT_REGISTRY.get(transport_type)
            if transport_cls is None:
                logger.debug(
                    "MCP server '%s': unknown transport '%s', skipping",
                    sname, transport_type,
                )
                self._unavailable.add(sname)
                continue
            transport = transport_cls(cfg)  # type: ignore[call-arg]
            try:
                transport.connect()
                self._transports[sname] = transport
                self._discover_tools(sname, transport)
                logger.info(
                    "MCP server '%s' connected via %s: %d tools",
                    sname, transport_type, len(self._tools),
                )
            except Exception as e:
                logger.debug("MCP server '%s' unavailable: %s", sname, e)
                self._unavailable.add(sname)
                try:
                    transport.disconnect()
                except Exception:
                    pass

    def _discover_tools(self, sname: str, transport: Transport) -> None:
        resp = transport.send_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {},
        })
        result = resp.get("result")
        if result is None:
            logger.warning("MCP tools/list returned no result for server '%s'", sname)
            return
        tools = result.get("tools", []) if isinstance(result, dict) else []
        for tool in tools:
            full_name = f"mcp__{sname}__{tool['name']}"
            self._tools[full_name] = MCPToolDef(
                name=tool["name"],
                description=tool.get("description", ""),
                input_schema=tool.get("inputSchema", {}),
            )

    def call_tool(
        self, server_name: str, tool_name: str, arguments: dict[str, Any],
        timeout: float | None = None,
    ) -> str:
        if server_name in self._unavailable:
            return (
                f"MCP server '{server_name}' is unavailable "
                f"(connection failed at startup). Try HTTP API fallback."
            )
        transport = self._transports.get(server_name)
        if not transport:
            return (
                f"MCP server '{server_name}' not connected. "
                f"Available servers: {list(self._transports.keys())}"
            )
        effective_timeout = timeout or self._servers.get(server_name, MCPServerConfig()).timeout
        resp = transport.send_request({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }, timeout=effective_timeout)
        if "error" in resp:
            return (
                f"MCP error from '{server_name}/{tool_name}': "
                f"{resp['error'].get('message', str(resp['error']))}"
            )
        result = resp.get("result", {})
        content = result.get("content", result)
        if isinstance(content, list):
            text_parts = [
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            return "\n".join(text_parts) if text_parts else json.dumps(content)
        return json.dumps(content) if not isinstance(content, str) else content

    def list_tool_definitions(self) -> list[dict[str, Any]]:
        """Return tool definitions in OpenAI function-calling format."""
        from agenthatch_core.tools.bus import _normalize_json_schema  # type: ignore[import-untyped]

        return [
            {
                "type": "function",
                "function": {
                    "name": full_name,
                    "description": t.description,
                    "parameters": _normalize_json_schema(t.input_schema),
                },
            }
            for full_name, t in self._tools.items()
        ]

    def register_with_capbus(self, capbus: Any) -> None:
        """Register all MCP tools as external handlers on CapBus."""
        from agenthatch_core.tools.bus import _normalize_json_schema  # type: ignore[import-untyped]

        for full_name, t in self._tools.items():
            sname = full_name.split("__", 1)[1]
            tool_name = t.name

            def make_handler(sn: str, tn: str) -> Callable[..., str]:
                def handler(**kwargs: Any) -> str:
                    return self.call_tool(sn, tn, kwargs)
                return handler

            normalized_schema = _normalize_json_schema(t.input_schema)
            capbus.register_external_tool(
                full_name, normalized_schema, make_handler(sname, tool_name)
            )

    def disconnect_all(self) -> None:
        for transport in self._transports.values():
            try:
                transport.disconnect()
            except Exception:
                pass
        self._transports.clear()
        self._unavailable.clear()

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    @property
    def unavailable_servers(self) -> set[str]:
        return self._unavailable.copy()
