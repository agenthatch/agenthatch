"""agenthatch 配置管理。

配置优先级：CLI 参数 > 环境变量 > 配置文件 > 默认值
"""
import os
import tomllib
from pathlib import Path
from typing import Any

from agenthatch.exceptions import ConfigError

CONFIG_DIR = Path.home() / ".agenthatch"
CONFIG_FILE = CONFIG_DIR / "config.toml"

_CONFIG_TEMPLATE = """\
# agenthatch 配置文件
# 详细说明: https://github.com/agenthatch/agenthatch

[core]
verbose = false

# LLM Provider 配置
# 注意：API Key 不要写在这里！请通过环境变量设置：
#   export OPENAI_API_KEY=sk-xxxx
#   export ANTHROPIC_API_KEY=sk-ant-xxxx
[providers]
default = "openai"
model = "gpt-4o"
"""


class Config:
    """配置加载器。"""

    @classmethod
    def load(
        cls,
        config_path: Path | None = None,
    ) -> dict[str, Any]:
        """加载配置（两级优先级）。

        Args:
            config_path: 自定义配置文件路径（None 则用默认）

        Returns:
            合并后的配置字典
        """
        config: dict[str, Any] = cls._load_file(config_path)
        config = cls._apply_env_overrides(config)
        return config

    @classmethod
    def _load_file(cls, config_path: Path | None = None) -> dict[str, Any]:
        """从 TOML 文件加载配置。"""
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
        """用 AGENTHATCH_* 环境变量覆盖配置。"""
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
        """生成默认配置文件。

        技术债：当前使用字符串模板写入，因为 tomllib 只读不写。
        后续版本需要频繁写入配置时引入 tomli-w。

        Args:
            force: 是否强制覆盖已有配置文件

        Returns:
            创建的配置文件路径

        Raises:
            ConfigError: 配置文件已存在且 force=False
        """
        if CONFIG_FILE.exists() and not force:
            raise ConfigError(
                f"Config file already exists: {CONFIG_FILE}\n"
                f"Use --force to overwrite."
            )
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(_CONFIG_TEMPLATE, encoding="utf-8")
        return CONFIG_FILE
