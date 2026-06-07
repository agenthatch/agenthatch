"""SandboxWhitelist — tiered command whitelist configuration.

Level 0 — replaces hardcoded _ALLOWED_COMMANDS with a configurable
dataclass.  Three tiers: NONE, STANDARD, EXTENDED.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agenthatch_core.bricks.manifest import SandboxTier

# Base set for STANDARD tier
STANDARD_COMMANDS: set[str] = {
    "python3", "python", "bash", "node", "curl", "jq",
    "cat", "head", "tail", "grep", "awk", "sed", "echo",
    "ls", "find", "wc", "sort", "uniq", "cut", "tr",
}

# Additional commands for EXTENDED tier
EXTENDED_COMMANDS: set[str] = {
    "pip", "pip3", "npm", "npx", "git", "docker", "make",
    "cargo", "go", "rustc", "javac", "java",
}


@dataclass
class SandboxWhitelist:
    """Tiered command whitelist for sandbox execution.

    Usage:
        whitelist = SandboxWhitelist.from_tier(SandboxTier.EXTENDED)
        whitelist.extend({"ffmpeg", "imagemagick"})
        allowed = "python3" in whitelist.commands  # True
    """

    tier: SandboxTier = SandboxTier.STANDARD
    commands: set[str] = field(default_factory=set)
    extra: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not self.commands:
            self.commands = self._tier_commands()

    def _tier_commands(self) -> set[str]:
        """Get base command set for the current tier."""
        if self.tier == SandboxTier.NONE:
            return set()
        if self.tier == SandboxTier.EXTENDED:
            return STANDARD_COMMANDS | EXTENDED_COMMANDS
        return set(STANDARD_COMMANDS)

    @classmethod
    def from_tier(cls, tier: SandboxTier) -> SandboxWhitelist:
        """Create whitelist from tier level."""
        return cls(tier=tier, commands=cls._tier_commands_static(tier))

    @staticmethod
    def _tier_commands_static(tier: SandboxTier) -> set[str]:
        if tier == SandboxTier.NONE:
            return set()
        if tier == SandboxTier.EXTENDED:
            return STANDARD_COMMANDS | EXTENDED_COMMANDS
        return set(STANDARD_COMMANDS)

    def extend(self, *commands: str) -> None:
        """Add extra commands to the whitelist."""
        self.extra.update(commands)
        self.commands.update(commands)

    def allows(self, command: str) -> bool:
        """Check if a command is allowed."""
        return command in self.commands