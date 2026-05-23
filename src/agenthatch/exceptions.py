"""agenthatch exception hierarchy.

v0.1: AgentHatchError, ConfigError
v0.2: + ProviderNotFoundError, ApiKeyError
"""


class AgentHatchError(Exception):
    """Base agenthatch exception."""
    exit_code = 1


class ConfigError(AgentHatchError):
    """Configuration file error."""
    exit_code = 2


class ProviderNotFoundError(AgentHatchError):
    """Requested provider not found in built-in registry or custom config.

    Raised when --provider references a name that is neither a built-in
    provider key nor a custom provider defined in config.toml.
    """
    exit_code = 2


class ApiKeyError(AgentHatchError):
    """API key is missing or failed connectivity verification.

    Raised when:
    - resolve_api_key returns None for a provider that requires a key
    - verify_api_key returns False (server returned 401/403)
    """
    exit_code = 1
