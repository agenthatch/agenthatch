"""Sandbox interface — stub for v0.4 runtime isolation.

v0.4 will provide Docker/podman-based sandbox execution for
SkillAgent runtime, with configurable runtime, timeout, and env.

v0.3 only defines the interface contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SandboxConfig:
    """Sandbox configuration consumed by v0.4 SkillAgent.from_ahspec()."""
    runtime: str | None = None   # python3.11, bash, node20, or None
    isolated: bool = False
    timeout: str = "60s"         # "15s", "60s", "120s", "600s"
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class Sandbox:
    """Sandbox stub for v0.4 runtime isolation.

    v0.4 will provide complete container-based sandbox execution.
    v0.3 only defines the config interface consumed by SkillAgent.from_ahspec().
    """

    config: SandboxConfig = field(default_factory=SandboxConfig)

    def setenv(self, name: str, value: str) -> None:
        """Set an environment variable in the sandbox. (v0.4 implementation)"""
        pass

    def run(self, command: str) -> str:
        """Execute a command in the sandbox. (v0.4 implementation)"""
        return ""

    def cleanup(self) -> None:
        """Clean up sandbox resources. (v0.4 implementation)"""
        pass
