"""Null stubs — zero-overhead replacements for disabled bricks.

Each null stub implements the same interface as its real counterpart
but does nothing.  This avoids conditionals scattered through the
agent loop — the loop calls the same methods regardless of whether
the brick is active.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class _NullCapBus:
    """No-op capability bus for prompt-only skills."""

    capabilities: dict[str, Any] = {}
    builtins: dict[str, Any] = {}
    unavailable: set[str] = set()
    _guard: Any = None
    _output_schemas: dict[str, Any] = {}

    def register(self, *args: Any, **kwargs: Any) -> None:
        pass

    def register_external_tool(self, *args: Any, **kwargs: Any) -> None:
        pass

    def inject_builtin(self, *args: Any, **kwargs: Any) -> None:
        pass

    def route(self, tool_name: str, arguments: dict[str, Any]) -> str:
        return f"Tool '{tool_name}' not available (CapBus disabled)."

    def list_tool_definitions(self) -> list[Any]:
        return []

    def mark_unavailable(self, *args: Any, **kwargs: Any) -> None:
        pass


class _NullSandbox:
    """No-op sandbox for skills that don't need command execution."""

    config: Any = None

    def configure(self, *args: Any, **kwargs: Any) -> None:
        pass

    def setenv(self, *args: Any, **kwargs: Any) -> None:
        pass

    def run(self, command: Any, **kwargs: Any) -> Any:
        cmd_str = " ".join(command) if isinstance(command, list) else str(command)
        return _NullSandboxResult(
            stderr=f"Sandbox disabled — cannot run '{cmd_str}'",
            returncode=1,
        )

    def cleanup(self) -> None:
        pass


class _NullSandboxResult:
    """No-op sandbox result."""
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    timed_out: bool = False

    def __init__(self, stderr: str = "", returncode: int = 0):
        self.stderr = stderr
        self.returncode = returncode


class _NullHooks:
    """No-op hooks manager."""

    def register(self, *args: Any, **kwargs: Any) -> None:
        pass

    def execute(self, point: Any, context: dict[str, Any]) -> dict[str, Any]:
        return context
