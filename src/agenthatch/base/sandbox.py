"""Sandbox — v0.4 subprocess-based isolated execution."""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass, field


@dataclass
class SandboxConfig:
    """Sandbox configuration."""
    runtime: str | None = None
    isolated: bool = False
    timeout: str = "60s"
    env: dict[str, str] = field(default_factory=dict)
    extra_allowed_commands: list[str] = field(default_factory=list)


@dataclass
class Sandbox:
    """Subprocess sandbox for capability execution."""

    config: SandboxConfig = field(default_factory=SandboxConfig)

    _allowed_commands: set[str] = field(default_factory=lambda: {
        "python3", "python", "bash", "node", "curl", "jq",
        "cat", "head", "tail", "grep", "awk", "sed", "echo",
        "ls", "find", "wc", "sort", "uniq", "cut", "tr",
    })

    def extend_whitelist(self, *commands: str) -> None:
        """Add commands to the sandbox whitelist."""
        self._allowed_commands.update(commands)

    def configure(
        self,
        runtime: str | None = None,
        isolated: bool | None = None,
        timeout: str | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        """Configure sandbox from AHSSPEC.base."""
        if runtime is not None:
            self.config.runtime = runtime
        if isolated is not None:
            self.config.isolated = isolated
        if timeout is not None:
            self.config.timeout = timeout
        if env is not None:
            self.config.env = env
        if self.config.extra_allowed_commands:
            self._allowed_commands.update(self.config.extra_allowed_commands)

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
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> str:
        """Execute command in sandbox with security checks."""
        merged_env = {**os.environ, **self.config.env}
        if env:
            merged_env.update(env)

        timeout_sec = timeout or self._parse_timeout(self.config.timeout)

        if isinstance(command, str):
            cmd_parts = shlex.split(command)
        else:
            cmd_parts = list(command)

        if not cmd_parts:
            return "Error: empty command"

        cmd_name = cmd_parts[0]
        if cmd_name not in self._allowed_commands:
            return (
                f"Error: command '{cmd_name}' is not in the sandbox whitelist. "
                f"Allowed: {', '.join(sorted(self._allowed_commands))}"
            )

        try:
            result = subprocess.run(
                cmd_parts,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                env=merged_env,
            )
            if result.returncode != 0:
                return (
                    f"Error (exit {result.returncode}): "
                    f"{result.stderr.strip() or result.stdout.strip() or '(no output)'}"
                )
            return result.stdout.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {timeout_sec}s"
        except FileNotFoundError:
            return f"Error: command '{cmd_name}' not found"
        except Exception as e:
            return f"Error: {e}"

    def cleanup(self) -> None:
        """Clean up sandbox resources."""
        pass
