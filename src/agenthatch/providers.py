"""Provider registry and API key resolution.

v0.2 single source of truth for all LLM providers.

Priority chain (highest to lowest):
1. Provider-specific env var    (e.g., OPENAI_API_KEY)
2. Generic env var              (AGENTHATCH_API_KEY)
3. Config file per-provider key ([providers.NAME].api_key)
4. Interactive Rich prompt      (tty only, password-masked)
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Literal

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore[no-redef,unused-ignore,import-not-found]
    except ImportError:
        tomllib = None  # type: ignore[assignment]

import httpx

from agenthatch.cli import console
from agenthatch.config import CONFIG_FILE
from agenthatch.exceptions import ProviderNotFoundError

# ---------------------------------------------------------------------------
# Built-in providers
# ---------------------------------------------------------------------------

_warned_providers: set[str] = set()

ProviderKind = Literal["builtin", "custom"]


@dataclass(frozen=True)
class ProviderFeatures:
    """Capability flags for a provider's API surface.

    Defaults assume OpenAI-compatible behavior (all features enabled).
    Custom providers can override specific flags via config.toml.
    """

    supports_tools: bool = True
    supports_stream_tools: bool = True
    supports_json_mode: bool = True
    supports_parallel_tool_calls: bool = True
    supports_reasoning_content: bool = False
    requires_anthropic_adapter: bool = False
    available_models: tuple[str, ...] = ()


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
    context_window: int = 128000
    features: ProviderFeatures = ProviderFeatures()


BUILTIN_PROVIDERS: dict[str, ProviderInfo] = {
    "openai": ProviderInfo(
        name="openai",
        kind="builtin",
        env_key="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o",
        context_window=128000,
        features=ProviderFeatures(
            supports_tools=True,
            supports_stream_tools=True,
            supports_json_mode=True,
            supports_parallel_tool_calls=True,
        ),
    ),
    "anthropic": ProviderInfo(
        name="anthropic",
        kind="builtin",
        env_key="ANTHROPIC_API_KEY",
        base_url="https://api.anthropic.com",
        default_model="claude-sonnet-4-20250514",
        context_window=200000,
        features=ProviderFeatures(
            supports_tools=True,
            supports_stream_tools=False,
            supports_json_mode=False,
            supports_parallel_tool_calls=True,
            supports_reasoning_content=True,
            requires_anthropic_adapter=True,
        ),
    ),
    "deepseek": ProviderInfo(
        name="deepseek",
        kind="builtin",
        env_key="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
        default_model="deepseek-chat",
        context_window=128000,
        features=ProviderFeatures(
            supports_tools=True,
            supports_stream_tools=True,
            supports_json_mode=True,
            supports_parallel_tool_calls=True,
            supports_reasoning_content=True,
            available_models=(
                "deepseek-chat",
                "deepseek-v4-flash",
                "deepseek-v4-pro",
            ),
        ),
    ),
    "ollama": ProviderInfo(
        name="ollama",
        kind="builtin",
        env_key="",
        base_url="http://localhost:11434/v1",
        default_model="llama3",
        context_window=4096,
        features=ProviderFeatures(
            supports_tools=False,
            supports_stream_tools=False,
            supports_json_mode=False,
            supports_parallel_tool_calls=False,
        ),
    ),
}

BUILTIN_PROVIDER_NAMES: frozenset[str] = frozenset(BUILTIN_PROVIDERS.keys())

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider capability probing
# ---------------------------------------------------------------------------


def _probe_reasoning_content(api_key: str, base_url: str, model: str) -> bool:
    """Probe whether provider returns reasoning_content with empty content."""
    try:
        import openai

        client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=10.0)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say hi in one word."}],
            max_tokens=20,
        )
        msg = resp.choices[0].message
        content = getattr(msg, "content", "")
        reasoning = getattr(msg, "reasoning_content", None)
        return bool(not content and reasoning)
    except Exception:
        return False


def _probe_provider_capabilities(
    api_key: str, base_url: str, model: str
) -> ProviderFeatures:
    """Probe a custom provider to auto-detect capabilities."""
    features = ProviderFeatures()

    try:
        import openai

        client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=10.0)

        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say hi."}],
            max_tokens=5,
        )
        msg = resp.choices[0].message
        if not getattr(msg, "content", "") and getattr(msg, "reasoning_content", None):
            features = ProviderFeatures(
                supports_reasoning_content=True,
                supports_tools=features.supports_tools,
                supports_stream_tools=features.supports_stream_tools,
                supports_json_mode=features.supports_json_mode,
            )

        try:
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "test"}],
                tools=[{"type": "function", "function": {
                    "name": "test", "parameters": {"type": "object", "properties": {}}
                }}],
                max_tokens=5,
            )
        except Exception:
            features = ProviderFeatures(
                supports_tools=False,
                supports_reasoning_content=features.supports_reasoning_content,
                supports_stream_tools=False,
                supports_json_mode=features.supports_json_mode,
            )

    except Exception as e:
        logger.warning("Provider probe failed for %s: %s", model, e)

    return features


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
    features_cfg = custom_section.get("features", {})
    if "supports_reasoning_content" not in features_cfg:
        api_key = custom_section.get("api_key", "")
        if api_key and _probe_reasoning_content(
            api_key=api_key,
            base_url=custom_section.get("base_url", ""),
            model=custom_section.get("default_model", ""),
        ):
            features_cfg["supports_reasoning_content"] = True
            logger.info(
                "Provider '%s': auto-detected reasoning_content support", name
            )
    features = ProviderFeatures(
        supports_tools=features_cfg.get("supports_tools", True),
        supports_stream_tools=features_cfg.get("supports_stream_tools", True),
        supports_json_mode=features_cfg.get("supports_json_mode", True),
        supports_parallel_tool_calls=features_cfg.get("supports_parallel_tool_calls", True),
        supports_reasoning_content=features_cfg.get("supports_reasoning_content", False),
        requires_anthropic_adapter=features_cfg.get("requires_anthropic_adapter", False),
        available_models=tuple(features_cfg.get("available_models", ())),
    )
    return ProviderInfo(
        name=name,
        kind="custom",
        env_key=custom_section.get("env_key", ""),
        base_url=custom_section.get("base_url", ""),
        default_model=custom_section.get("default_model", ""),
        features=features,
    )


def get_default_provider(config: dict[str, Any] | None = None) -> str:
    """Return the default provider name from config."""
    if config is None:
        config = _load_config_safe()
    providers_section: dict[str, Any] = config.get("providers", {})
    return str(providers_section.get("default", "openai"))


# ---------------------------------------------------------------------------
# API Key resolution
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

    # NOTE: Builtin providers without env_key (e.g. Ollama) don't need API keys
    # https://github.com/agenthatch/agenthatch/issues
    if info.kind == "builtin" and not info.env_key:
        return "local-no-key"

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
        if provider not in _warned_providers:
            _warned_providers.add(provider)
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
            features_cfg = cfg.get("features", {})
            features = ProviderFeatures(
                supports_tools=features_cfg.get("supports_tools", True),
                supports_stream_tools=features_cfg.get("supports_stream_tools", True),
                supports_json_mode=features_cfg.get("supports_json_mode", True),
                supports_parallel_tool_calls=features_cfg.get("supports_parallel_tool_calls", True),
                supports_reasoning_content=features_cfg.get("supports_reasoning_content", False),
                requires_anthropic_adapter=features_cfg.get("requires_anthropic_adapter", False),
                available_models=tuple(features_cfg.get("available_models", ())),
            )
            result.append(
                ProviderInfo(
                    name=f"custom.{name}",
                    kind="custom",
                    env_key=cfg.get("env_key", ""),
                    base_url=cfg.get("base_url", ""),
                    default_model=cfg.get("default_model", ""),
                    features=features,
                )
            )
    return sorted(result, key=lambda p: p.name)


def list_all_providers(config: dict[str, Any] | None = None) -> list[ProviderInfo]:
    """Return all providers (built-in + custom)."""
    return list_builtin_providers() + list_custom_providers(config)
