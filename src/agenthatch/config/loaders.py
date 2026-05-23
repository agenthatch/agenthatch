"""Provider configuration loader.

Loaded config validators for multi-provider configurations.
v0.1 was a placeholder — v0.2 activates this module.
"""

from __future__ import annotations

from typing import Any

from agenthatch.exceptions import ConfigError
from agenthatch.providers import BUILTIN_PROVIDER_NAMES, ProviderInfo

__all__ = ["load_provider_config", "resolve_provider_section"]


def load_provider_config(config: dict[str, Any], provider: str) -> dict[str, Any]:
    """Load the configuration section for a specific provider.

    Works for both built-in and custom providers.

    Args:
        config: Full loaded configuration dict
        provider: Provider name ("openai", "custom.my-llm", etc.)

    Returns:
        Provider section dict with keys: api_key, base_url, default_model

    Raises:
        ConfigError: Provider section missing or malformed
    """
    section = resolve_provider_section(config, provider)
    return _normalize_provider_section(section, provider)


def resolve_provider_section(config: dict[str, Any], provider: str) -> dict[str, Any]:
    """Resolve the raw TOML section for a provider.

    Built-in providers live at [providers.<name>].
    Custom providers live at [providers.custom.<name>].

    Returns:
        Raw provider section from TOML

    Raises:
        ConfigError: Section not found or not a dict
    """
    providers_section = config.get("providers")
    if not isinstance(providers_section, dict):
        raise ConfigError("[providers] section is missing or not a table")

    if provider.startswith("custom."):
        custom_key = provider.removeprefix("custom.")
        section = providers_section.get("custom", {}).get(custom_key)
    else:
        section = providers_section.get(provider)

    if not isinstance(section, dict):
        raise ConfigError(
            f"Provider '{provider}' not configured in config.toml"
        )
    return section


def _normalize_provider_section(
    section: dict[str, Any], provider: str
) -> dict[str, Any]:
    """Ensure required keys exist with sensible defaults.

    Fills in missing keys from BUILTIN_PROVIDERS if the provider is built-in.
    """
    result = dict(section)

    if provider in BUILTIN_PROVIDER_NAMES:
        from agenthatch.providers import BUILTIN_PROVIDERS

        builtin: ProviderInfo = BUILTIN_PROVIDERS[provider]
        result.setdefault("base_url", builtin.base_url)
        result.setdefault("env_key", builtin.env_key)
        result.setdefault("default_model", builtin.default_model)

    result.setdefault("api_key", "")
    if "base_url" not in result:
        raise ConfigError(
            f"Custom provider '{provider}' must specify 'base_url' in config.toml"
        )
    result.setdefault("default_model", "")
    return result
