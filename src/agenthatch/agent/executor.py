"""CapabilityExecutor — wraps CapBus.route for sandboxed execution (v0.4)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass
class ExecutionResult:
    """Result of executing a capability."""
    success: bool
    output: str
    elapsed_ms: float = 0.0
    error: str | None = None


class CapabilityExecutor:
    """Wraps CapBus.route with timing and error handling."""

    def __init__(self, capbus: Any):
        self._capbus = capbus

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> ExecutionResult:
        """Execute a tool call through CapBus."""
        t0 = time.time()
        try:
            result = self._capbus.route(tool_name, arguments)
            elapsed = (time.time() - t0) * 1000
            return ExecutionResult(
                success=True,
                output=str(result),
                elapsed_ms=elapsed,
            )
        except Exception as e:
            elapsed = (time.time() - t0) * 1000
            return ExecutionResult(
                success=False,
                output="",
                elapsed_ms=elapsed,
                error=str(e),
            )
