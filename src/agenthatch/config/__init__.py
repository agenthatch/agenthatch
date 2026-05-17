"""agenthatch configuration management.

Configuration priority: CLI args > environment variables > config file > defaults
"""
import os
import tomllib
from pathlib import Path
from typing import Any

from agenthatch.exceptions import ConfigError

CONFIG_DIR = Path.home() / ".agenthatch"
CONFIG_FILE = CONFIG_DIR / "config.toml"

_CONFIG_TEMPLATE = """\
# agenthatch configuration file
# Docs: https://github.com/agenthatch/agenthatch

[core]
verbose = false

# LLM Provider configuration
# IMPORTANT: Do not put API keys here! Set them via environment variables:
#   export OPENAI_API_KEY=sk-xxxx
#   export ANTHROPIC_API_KEY=sk-ant-xxxx
[providers]
default = "openai"
model = "gpt-4o"
"""


class Config:
    """Configuration loader."""

    @classmethod
    def load(
        cls,
        config_path: Path | None = None,
    ) -> dict[str, Any]:
        """Load configuration (two-level priority).

        Args:
            config_path: Custom config file path (None uses default)

        Returns:
            Merged configuration dictionary
        """
        config: dict[str, Any] = cls._load_file(config_path)
        config = cls._apply_env_overrides(config)
        return config

    @classmethod
    def _load_file(cls, config_path: Path | None = None) -> dict[str, Any]:
        """Load configuration from a TOML file."""
        path = config_path or CONFIG_FILE
        if not path.exists():
            return {}
        try:
            with open(path, "rb") as f:
                return tomllib.load(f)
        except (tomllib.TOMLDecodeError, OSError) as e:
            raise ConfigError(f"Failed to read config file {path}: {e}") from e

    @classmethod
    def _apply_env_overrides(cls, config: dict[str, Any]) -> dict[str, Any]:
        """Override configuration with AGENTHATCH_* environment variables."""
        env_map: dict[str, tuple[str, str]] = {
            "AGENTHATCH_VERBOSE": ("core", "verbose"),
            "AGENTHATCH_PROVIDER": ("providers", "default"),
            "AGENTHATCH_MODEL": ("providers", "model"),
        }
        _bool_keys: set[str] = {"verbose"}
        for env_key, (section, key) in env_map.items():
            value = os.environ.get(env_key)
            if value is not None:
                if key in _bool_keys:
                    config.setdefault(section, {})[key] = value.lower() in ("true", "1", "yes")
                else:
                    config.setdefault(section, {})[key] = value
        return config

    @classmethod
    def create_default(cls, force: bool = False) -> Path:
        """Create a default configuration file.

        Tech debt: currently uses string template for writing because tomllib
        is read-only. Switch to tomli-w when frequent config writes are needed.

        Args:
            force: Whether to force overwrite an existing config file

        Returns:
            Path to the created config file

        Raises:
            ConfigError: Config file already exists and force=False
        """
        if CONFIG_FILE.exists() and not force:
            raise ConfigError(
                f"Config file already exists: {CONFIG_FILE}\n"
                f"Use --force to overwrite."
            )
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(_CONFIG_TEMPLATE, encoding="utf-8")
        return CONFIG_FILE
