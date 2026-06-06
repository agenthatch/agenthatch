"""Sandbox — subprocess-based isolated execution (agenthatch-core)."""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SandboxConfig:
    """Sandbox configuration."""
    runtime: str | None = None
    isolated: bool = False
    timeout: str = "60s"
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class SandboxResult:
    """Result of sandbox execution."""
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    timed_out: bool = False


@dataclass
class Sandbox:
    """Subprocess sandbox for capability execution."""

    config: SandboxConfig = field(default_factory=SandboxConfig)

    _ALLOWED_COMMANDS: set[str] = field(default_factory=lambda: {
        "python3", "python", "bash", "node", "curl", "jq",
        "cat", "head", "tail", "grep", "awk", "sed", "echo",
        "ls", "find", "wc", "sort", "uniq", "cut", "tr",
    })

    def configure(
        self,
        runtime: str | None = None,
        isolated: bool | None = None,
        timeout: str | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        """Configure sandbox from spec."""
        if runtime is not None:
            self.config.runtime = runtime
        if isolated is not None:
            self.config.isolated = isolated
        if timeout is not None:
            self.config.timeout = timeout
        if env is not None:
            self.config.env = env

    def setenv(self, name: str, value: str) -> None:
        """Set an environment variable."""
        self.config.env[name] = value

    def _parse_timeout(self, timeout_str: str) -> int:
        """Parse timeout string like '60s' to seconds."""
        s = timeout_str.rstrip("s")
        try:
            return int(s)
        except ValueError:
            return 60

    def run(
        self,
        command: str | list[str],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> SandboxResult:
        """Execute command in sandbox with security checks.

        Returns SandboxResult with stdout, stderr, returncode, and timed_out.
        """
        merged_env = {**os.environ, **self.config.env}
        if env:
            merged_env.update(env)

        timeout_sec = timeout or self._parse_timeout(self.config.timeout)

        if isinstance(command, str):
            cmd_parts = shlex.split(command)
        else:
            cmd_parts = list(command)

        if not cmd_parts:
            return SandboxResult(stderr="Error: empty command", returncode=1)

        cmd_name = cmd_parts[0]
        if cmd_name not in self._ALLOWED_COMMANDS:
            return SandboxResult(
                stderr=(
                    f"Error: command '{cmd_name}' is not in the sandbox whitelist. "
                    f"Allowed: {', '.join(sorted(self._ALLOWED_COMMANDS))}"
                ),
                returncode=1,
            )

        try:
            result = subprocess.run(
                cmd_parts,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                env=merged_env,
                cwd=cwd,
            )
            return SandboxResult(
                stdout=result.stdout.strip() or "(no output)",
                stderr=result.stderr.strip(),
                returncode=result.returncode,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                stderr=f"Error: command timed out after {timeout_sec}s",
                returncode=1,
                timed_out=True,
            )
        except FileNotFoundError:
            return SandboxResult(
                stderr=f"Error: command '{cmd_name}' not found",
                returncode=1,
            )
        except Exception as e:
            return SandboxResult(stderr=f"Error: {e}", returncode=1)

    def cleanup(self) -> None:
        """Clean up sandbox resources."""
        pass