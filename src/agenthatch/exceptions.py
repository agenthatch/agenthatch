"""agenthatch exception hierarchy.

v0.1 defines only two base exceptions.
Future versions will expand as needed (SkillParseError → v0.2, AgentNotFoundError → v0.4).
"""


class AgentHatchError(Exception):
    """Base agenthatch exception."""
    exit_code = 1


class ConfigError(AgentHatchError):
    """Configuration file error."""
    exit_code = 2
