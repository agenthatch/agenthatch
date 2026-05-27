"""Provider registry and API key resolution.

v0.2 single source of truth for all LLM providers.
References: Codex CLI auth/manager.rs, Claude Code utils/auth.ts.

Priority chain (highest to lowest):
1. Provider-specific env var    (e.g., OPENAI_API_KEY)
2. Generic env var              (AGENTHATCH_API_KEY)
3. Config file per-provider key ([providers.NAME].api_key)
4. Interactive Rich prompt      (tty only, password-masked)
"""

from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from agenthatch.cli import console
from agenthatch.config import CONFIG_FILE
from agenthatch.exceptions import ProviderNotFoundError

# ---------------------------------------------------------------------------
# Built-in providers (Codex CLI pattern: ModelProviderInfo in config_toml.rs)
# ---------------------------------------------------------------------------

ProviderKind = Literal["builtin", "custom"]


@dataclass(frozen=True)
class ProviderInfo:
    """Metadata for one LLM provider.

    Built-in providers use frozen defaults. Custom providers are loaded
    from config.toml [providers.custom.*] sections.
    """

    name: str
    kind: ProviderKind = "builtin"
    env_key: str = ""
    base_url: str = ""
    default_model: str = ""


BUILTIN_PROVIDERS: dict[str, ProviderInfo] = {
    "openai": ProviderInfo(
        name="openai",
        kind="builtin",
        env_key="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o",
    ),
    "anthropic": ProviderInfo(
        name="anthropic",
        kind="builtin",
        env_key="ANTHROPIC_API_KEY",
        base_url="https://api.anthropic.com",
        default_model="claude-sonnet-4-20250514",
    ),
    "deepseek": ProviderInfo(
        name="deepseek",
        kind="builtin",
        env_key="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
        default_model="deepseek-chat",
    ),
    "ollama": ProviderInfo(
        name="ollama",
        kind="builtin",
        env_key="",  # local — no API key required
        base_url="http://localhost:11434/v1",
        default_model="llama3",
    ),
}

BUILTIN_PROVIDER_NAMES: frozenset[str] = frozenset(BUILTIN_PROVIDERS.keys())


# ---------------------------------------------------------------------------
# Provider resolution
# ---------------------------------------------------------------------------

def get_provider(name: str, config: dict[str, Any] | None = None) -> ProviderInfo:
    """Resolve a provider by name.

    Order:
    1. Built-in providers (openai, anthropic, deepseek, ollama)
    2. Custom providers from config.toml [providers.custom.NAME]

    Args:
        name: Provider name (e.g., "openai", "custom.my-llm")
        config: Loaded configuration dict. If None, loads from disk.

    Returns:
        ProviderInfo for the matched provider.

    Raises:
        ProviderNotFoundError: Provider not found in built-in or custom.
    """
    if name in BUILTIN_PROVIDERS:
        return BUILTIN_PROVIDERS[name]

    if name.startswith("custom."):
        if config is None:
            config = _load_config_safe()
        return _resolve_custom_provider(name, config)

    raise ProviderNotFoundError(
        f"Unknown provider: {name}\n"
        f"Built-in providers: {', '.join(sorted(BUILTIN_PROVIDER_NAMES))}\n"
        f"Custom providers use 'custom.<name>' format."
    )


def _resolve_custom_provider(name: str, config: dict[str, Any]) -> ProviderInfo:
    """Resolve a custom provider from configuration."""
    custom_key = name.removeprefix("custom.")
    custom_section = config.get("providers", {}).get("custom", {}).get(custom_key)
    if not custom_section or not isinstance(custom_section, dict):
        raise ProviderNotFoundError(
            f"Custom provider '{name}' not found in config.\n"
            f"Add [providers.custom.{custom_key}] to {CONFIG_FILE}"
        )
    return ProviderInfo(
        name=name,
        kind="custom",
        env_key=custom_section.get("env_key", ""),
        base_url=custom_section.get("base_url", ""),
        default_model=custom_section.get("default_model", ""),
    )


def get_default_provider(config: dict[str, Any] | None = None) -> str:
    """Return the default provider name from config."""
    if config is None:
        config = _load_config_safe()
    providers_section: dict[str, Any] = config.get("providers", {})
    return str(providers_section.get("default", "openai"))


# ---------------------------------------------------------------------------
# API Key resolution (Claude Code pattern: getAnthropicApiKeyWithSource)
# ---------------------------------------------------------------------------

def resolve_api_key(
    provider: str,
    config: dict[str, Any] | None = None,
    prompt: bool = True,
) -> str | None:
    """Resolve API key using the 4-level priority chain.

    1. Provider-specific env var (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
    2. Generic env var (AGENTHATCH_API_KEY)
    3. Config file per-provider api_key field ([providers.NAME].api_key)
    4. Interactive Rich prompt (tty only, password-masked)

    Args:
        provider: Provider name (e.g., "openai", "custom.my-llm")
        config: Loaded config dict. None loads from disk.
        prompt: If False, skip interactive prompt (used in CI/non-tty).

    Returns:
        API key string, or None if no key could be resolved.

    Raises:
        ProviderNotFoundError: Provider name not recognized.
    """
    info = get_provider(provider, config)
    if config is None:
        config = _load_config_safe()

    # Level 1: Provider-specific env var
    if info.env_key:
        key = _read_env_var(info.env_key)
        if key:
            return key

    # Level 2: Generic env var
    key = _read_env_var("AGENTHATCH_API_KEY")
    if key:
        return key

    # Level 3: Config file
    key = _read_config_key(provider, config)
    if key:
        console.print(
            f"[yellow]Warning: API key for '{provider}' read from config file.[/yellow]",
        )
        console.print(
            f"[yellow]  Consider using {info.env_key} environment variable instead.[/yellow]",
        )
        return key

    # Level 4: Interactive prompt
    if prompt and sys.stdout.isatty():
        from rich.prompt import Prompt

        console.print(f"[accent]No API key found for '{provider}'.[/accent]")
        key = Prompt.ask(
            f"Enter API key for {provider}",
            password=True,
        )
        if key.strip():
            return key.strip()

    return None


def _read_env_var(name: str) -> str | None:
    """Read and trim an environment variable."""
    value = os.environ.get(name)
    if value and value.strip():
        return value.strip()
    return None


def _read_config_key(provider: str, config: dict[str, Any]) -> str | None:
    """Read api_key from config file for a given provider."""
    if provider.startswith("custom."):
        custom_key = provider.removeprefix("custom.")
        key = config.get("providers", {}).get("custom", {}).get(custom_key, {}).get("api_key")
    else:
        key = config.get("providers", {}).get(provider, {}).get("api_key")
    if isinstance(key, str) and key.strip():
        return key.strip()
    return None


def _load_config_safe() -> dict[str, Any]:
    """Load config file, returning empty dict on failure."""
    if not CONFIG_FILE.exists():
        return {}
    try:
        with open(CONFIG_FILE, "rb") as f:
            return tomllib.load(f)
    except (tomllib.TOMLDecodeError, OSError):
        return {}


# ---------------------------------------------------------------------------
# API Key verification (lightweight connectivity check)
# ---------------------------------------------------------------------------

def verify_api_key(
    provider: str,
    api_key: str,
    base_url: str,
    timeout: float = 10.0,
) -> tuple[bool, str]:
    """Verify an API key by making a lightweight HTTP request.

    Strategy:
    - GET {base_url}[/v1]/models with appropriate auth header
    - OpenAI-compatible providers: Bearer token auth
    - Anthropic: x-api-key header auth
    - 200/2xx = key is valid
    - 401/403 = key is invalid
    - Timeout/connection error = cannot verify (treated as uncertain, not failure)

    Returns:
        (ok, detail) tuple:
        - ok=True: key verified successfully
        - ok=False: key verification failed (detail explains why)
    """
    try:
        # Build auth headers: Anthropic uses x-api-key, others use Bearer
        auth_headers = (
            {"x-api-key": api_key}
            if provider == "anthropic"
            else {"Authorization": f"Bearer {api_key}"}
        )
        # Build models endpoint: ensure /v1 prefix for providers that need it
        stripped = base_url.rstrip("/")
        if "/v1" in stripped:
            url = f"{stripped}/models"
        else:
            url = f"{stripped}/v1/models"

        r = httpx.get(
            url,
            headers=auth_headers,
            timeout=timeout,
            follow_redirects=True,
        )
        if r.is_success:
            return True, f"Connected to {provider} ({r.status_code})"
        if r.status_code in (401, 403):
            return False, f"Authentication failed ({r.status_code}) — check API key"
        return False, f"Unexpected response ({r.status_code})"
    except httpx.TimeoutException:
        return True, f"Connection to {provider} timed out — key format accepted"
    except httpx.ConnectError:
        return True, f"Cannot reach {provider} — key not verified"
    except Exception as e:
        return True, f"Verification skipped: {e}"


# ---------------------------------------------------------------------------
# Provider listing
# ---------------------------------------------------------------------------

def list_builtin_providers() -> list[ProviderInfo]:
    """Return all built-in providers in registry order."""
    return list(BUILTIN_PROVIDERS.values())


def list_custom_providers(config: dict[str, Any] | None = None) -> list[ProviderInfo]:
    """Return all custom providers from config."""
    if config is None:
        config = _load_config_safe()
    result: list[ProviderInfo] = []
    custom_section = config.get("providers", {}).get("custom", {})
    if not isinstance(custom_section, dict):
        return result
    for name, cfg in custom_section.items():
        if isinstance(cfg, dict):
            result.append(
                ProviderInfo(
                    name=f"custom.{name}",
                    kind="custom",
                    env_key=cfg.get("env_key", ""),
                    base_url=cfg.get("base_url", ""),
                    default_model=cfg.get("default_model", ""),
                )
            )
    return sorted(result, key=lambda p: p.name)


def list_all_providers(config: dict[str, Any] | None = None) -> list[ProviderInfo]:
    """Return all providers (built-in + custom)."""
    return list_builtin_providers() + list_custom_providers(config)
