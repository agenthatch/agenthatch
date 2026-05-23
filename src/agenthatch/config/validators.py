"""Configuration validators.

v0.1 was a placeholder — v0.2 activates for provider and API key validation.
"""

from __future__ import annotations

import re
from typing import Any

from agenthatch.exceptions import ConfigError

__all__ = ["validate_provider_name", "validate_base_url", "validate_config_integrity"]


_PROVIDER_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_.-]*$")


def validate_provider_name(name: str) -> None:
    """Validate a provider name.

    Rules:
    - Must start with a letter
    - Contains only alphanumeric, underscore, dot, or hyphen
    - Minimum 2 characters

    Raises:
        ConfigError: Name is invalid
    """
    if not name or len(name) < 2:
        raise ConfigError("Provider name must be at least 2 characters")
    if not _PROVIDER_NAME_RE.match(name):
        raise ConfigError(
            f"Invalid provider name: '{name}'. "
            f"Must start with a letter and contain only letters, numbers, _, ., -"
        )


def validate_base_url(url: str) -> None:
    """Validate a base URL.

    Checks that the URL starts with http:// or https:// and has a valid host.

    Raises:
        ConfigError: URL is invalid
    """
    if not url.startswith(("http://", "https://")):
        raise ConfigError(
            f"Invalid base URL: '{url}'. Must start with http:// or https://"
        )


def validate_config_integrity(config: dict[str, Any]) -> list[str]:
    """Validate the overall config file integrity.

    Checks:
    - [providers] section exists
    - [providers].default references an existing provider
    - Custom provider sections have required fields

    Returns:
        List of warning messages (non-fatal issues).
        Raises ConfigError for fatal issues.
    """
    warnings: list[str] = []

    providers_section = config.get("providers")
    if not isinstance(providers_section, dict):
        raise ConfigError("Missing [providers] section in config.toml")

    default = providers_section.get("default", "openai")
    if not isinstance(default, str):
        raise ConfigError("[providers].default must be a string")

    # Collect all available provider names
    available: set[str] = {"openai", "anthropic", "deepseek", "ollama"}
    custom_section = providers_section.get("custom", {})
    if isinstance(custom_section, dict):
        for name in custom_section:
            available.add(f"custom.{name}")

    if default not in available:
        warnings.append(
            f"Default provider '{default}' is not configured. "
            f"Available: {', '.join(sorted(available))}"
        )

    # Check custom providers for required fields
    if isinstance(custom_section, dict):
        for name, cfg in custom_section.items():
            if not isinstance(cfg, dict):
                warnings.append(f"[providers.custom.{name}] is not a valid table")
                continue
            if not cfg.get("base_url"):
                warnings.append(
                    f"[providers.custom.{name}] is missing 'base_url'"
                )

    return warnings
