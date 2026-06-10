"""agenthatch configuration management.

Configuration priority: CLI args > environment variables > config file > defaults
"""
import os
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[import-not-found,no-redef]

from agenthatch.exceptions import ConfigError

CONFIG_DIR = Path.home() / ".agenthatch"
CONFIG_FILE = CONFIG_DIR / "config.toml"

_CONFIG_TEMPLATE = """\
# agenthatch configuration file
# Docs: https://github.com/agenthatch/agenthatch

[core]
verbose = false

# Default LLM provider (openai, anthropic, deepseek, ollama, or custom.<name>)
[providers]
default = "openai"

# OpenAI
# API key: set via environment variable OPENAI_API_KEY
[providers.openai]
api_key = ""
base_url = "https://api.openai.com/v1"
default_model = "gpt-4o"

# Anthropic
# API key: set via environment variable ANTHROPIC_API_KEY
[providers.anthropic]
api_key = ""
base_url = "https://api.anthropic.com"
default_model = "claude-sonnet-4-20250514"

# DeepSeek
# API key: set via environment variable DEEPSEEK_API_KEY
[providers.deepseek]
api_key = ""
base_url = "https://api.deepseek.com/v1"
default_model = "deepseek-chat"

# Ollama (local — no API key needed)
[providers.ollama]
api_key = ""
base_url = "http://localhost:11434/v1"
default_model = "llama3"

# Custom OpenAI-compatible providers
# Add your own under [providers.custom.<name>]
# [providers.custom.my-llm]
# api_key = ""
# base_url = "http://localhost:8000/v1"
# default_model = "mixtral-8x7b"
#
# v0.4: Capability flags (omit to use defaults = all True)
# [providers.custom.my-llm.features]
# supports_stream_tools = false
# supports_parallel_tool_calls = false

# ── v0.3: Skillhouse index ──
[skillhouse]
# Path to skillhouse.json (relative or absolute)
path = ".agenthatch/skillhouse.json"
# Auto-register skills to index after hatch
auto_index = true
# Embedding model for semantic search (sentence-transformers)
embedding_model = "all-MiniLM-L6-v2"

# ── v0.3: Skill search directories ──
[skills]
# Global skill search roots (comma-separated, for skill discovery)
search_dirs = "~/.agenthatch/skills,./skills,~/skills"

# ── v0.3 enhancement: Hatch file reading ──
[hatch]
# Max chars per-file when reading skill directory contents
max_file_chars = 10000
# Max file size in bytes (files larger are skipped)
max_file_bytes = 1000000
# Set to false to only collect file metadata (not contents)
read_file_contents = true

# ── v0.3: Harness model tiers ──
[harness]
# Large model (for Harness C and D — interface inference + base detection)
# Leave empty to fall back to provider's default_model
large_model = ""

# Small model (for Harness A, B, E — identity, intent, assembly)
# Leave empty to fall back to provider's default_model
small_model = ""
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
            "AGENTHATCH_LLM_PROVIDER": ("providers", "default"),
            # v0.3 extensions
            "AGENTHATCH_SKILLHOUSE_PATH": ("skillhouse", "path"),
            "AGENTHATCH_SKILL_DIRS": ("skills", "search_dirs"),
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
    def create_default(
        cls,
        force: bool = False,
        provider: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        env_key: str | None = None,
    ) -> Path:
        """Create a default configuration file.

        Tech debt: currently uses string template for writing because tomllib
        is read-only. Switch to tomli-w when frequent config writes are needed.

        The optional provider parameters provide a programmatic API for
        non-interactive config creation (e.g. from Python scripts or CI).
        The interactive CLI flow (agenthatch init) bypasses this method
        and writes config directly via _write_multi_provider_config.

        Args:
            force: Whether to force overwrite an existing config file
            provider: Default provider name
            api_key: API key to store (env var recommended instead)
            model: Default model ID
            base_url: Custom base URL (for custom providers)
            env_key: Custom env var name for API key (for custom providers)

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

        template = _CONFIG_TEMPLATE

        if provider and provider != "openai":
            template = template.replace(
                'default = "openai"',
                f'default = "{provider}"',
            )

        CONFIG_FILE.write_text(template, encoding="utf-8")
        return CONFIG_FILE
