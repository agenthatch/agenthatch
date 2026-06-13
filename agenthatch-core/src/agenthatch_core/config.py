"""Runtime configuration resolver (agenthatch-core).

Supports ${VAR} and ${VAR:-default} expansion for environment variables.
Detects plaintext API keys and issues warnings.
Provides API key auto-inheritance from agenthatch global config.
"""

import os
import re
import tomllib
from pathlib import Path
from typing import Any


_VAR_PATTERN = re.compile(r'\$\{(\w+)(?::-([^}]*))?\}')


def resolve_runtime_config(raw: dict) -> dict:
    """Recursively resolve ${VAR} in runtime config.

    Patterns supported:
    - ${VAR} → os.environ.get(VAR, "")
    - ${VAR:-default} → os.environ.get(VAR, "default")
    """
    def _resolve(value: Any) -> Any:
        if not isinstance(value, str):
            return value

        def _replacer(m: re.Match) -> str:
            var_name = m.group(1)
            default = m.group(2)
            return os.environ.get(var_name, default or "")

        return _VAR_PATTERN.sub(_replacer, value)

    def _walk(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_walk(v) for v in obj]
        return _resolve(obj)

    result = _walk(raw)

    # Check for plaintext API keys and warn
    _warn_plaintext_api_key(result)
    return result


def _warn_plaintext_api_key(result: dict) -> None:
    """Warn if api_key appears to be a plaintext secret."""
    if "llm" not in result:
        return

    llm_cfg = result["llm"]
    if not isinstance(llm_cfg, dict):
        return

    api_key = llm_cfg.get("api_key")
    if (
        api_key is not None
        and isinstance(api_key, str)
        and len(api_key) > 20
        and not api_key.startswith("$")
        and not api_key.startswith("{")
    ):
        import warnings
        warnings.warn(
            "api_key appears to be a plaintext secret. "
            "Use ${PROVIDER_API_KEY} syntax instead to read from environment.",
            UserWarning,
            stacklevel=2,
        )


def _resolve_provider_from_config(config: dict[str, Any], provider: str) -> dict[str, Any]:
    """Resolve provider config handling custom.xxx nested keys.

    v0.8.17: Previously used a flat .get() which failed for
    'custom.hermes' because it's stored as providers.custom.hermes.
    """
    providers = config.get("providers", {})
    if not isinstance(providers, dict):
        return {}
    if provider.startswith("custom."):
        custom_key = provider.removeprefix("custom.")
        return providers.get("custom", {}).get(custom_key, {})
    return providers.get(provider, {})


def inherit_api_key(config: dict[str, Any]) -> dict[str, Any]:
    """Fill missing API key and base_url from agenthatch global config.

    Reads ~/.agenthatch/config.toml (standard path, no agenthatch import).
    Only fills if the key is missing or empty — never overwrites explicit keys.

    This function lives in agenthatch-core (no agenthatch dependency),
    compliant with ADR-09 (one-way dependency).
    """
    llm_cfg = config.get("llm", {})
    if not isinstance(llm_cfg, dict):
        return config

    provider = llm_cfg.get("provider", "deepseek")

    # Read agenthatch global config (standard path, no agenthatch dependency)
    ah_config_path = Path.home() / ".agenthatch" / "config.toml"
    if not ah_config_path.exists():
        return config

    try:
        ah_cfg = tomllib.loads(ah_config_path.read_text())
    except Exception:
        return config

    provider_cfg = _resolve_provider_from_config(ah_cfg, provider)
    if not isinstance(provider_cfg, dict):
        return config

    # Inherit api_key if missing
    api_key = llm_cfg.get("api_key", "")
    if not api_key or not api_key.strip():
        ah_api_key = provider_cfg.get("api_key", "")
        if ah_api_key and isinstance(ah_api_key, str) and not ah_api_key.startswith("${"):
            llm_cfg["api_key"] = ah_api_key

    # Inherit base_url if missing (critical: otherwise defaults to openai.com)
    base_url = llm_cfg.get("base_url", "")
    if not base_url:
        ah_base_url = provider_cfg.get("base_url", "")
        if ah_base_url:
            llm_cfg["base_url"] = ah_base_url

    return config
