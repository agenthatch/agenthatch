"""Test config loaders (v0.2)."""

from __future__ import annotations

import pytest

from agenthatch.config.loaders import (
    _normalize_provider_section,
    load_provider_config,
    resolve_provider_section,
)
from agenthatch.exceptions import ConfigError


class TestResolveProviderSection:
    """resolve_provider_section tests."""

    def test_builtin_provider(self):
        config = {
            "providers": {
                "default": "openai",
                "openai": {"api_key": "sk-test", "base_url": "https://api.openai.com/v1"},
            }
        }
        section = resolve_provider_section(config, "openai")
        assert section["api_key"] == "sk-test"
        assert section["base_url"] == "https://api.openai.com/v1"

    def test_custom_provider(self):
        config = {
            "providers": {
                "default": "custom.my-llm",
                "custom": {
                    "my-llm": {"api_key": "key123", "base_url": "http://localhost:8000/v1"},
                },
            }
        }
        section = resolve_provider_section(config, "custom.my-llm")
        assert section["api_key"] == "key123"

    def test_missing_providers_section(self):
        with pytest.raises(ConfigError, match="providers"):
            resolve_provider_section({}, "openai")

    def test_providers_section_not_dict(self):
        with pytest.raises(ConfigError, match="providers"):
            resolve_provider_section({"providers": "invalid"}, "openai")

    def test_provider_not_configured(self):
        config = {"providers": {"default": "openai"}}
        with pytest.raises(ConfigError, match="not configured"):
            resolve_provider_section(config, "openai")

    def test_custom_provider_not_configured(self):
        config = {"providers": {"default": "openai", "custom": {}}}
        with pytest.raises(ConfigError, match="not configured"):
            resolve_provider_section(config, "custom.missing")


class TestNormalizeProviderSection:
    """_normalize_provider_section tests."""

    def test_builtin_fills_defaults(self):
        section = {"api_key": "sk-key"}
        result = _normalize_provider_section(section, "openai")
        assert result["api_key"] == "sk-key"
        assert result["base_url"] == "https://api.openai.com/v1"
        assert result["default_model"] == "gpt-4o"

    def test_ollama_no_env_key(self):
        section = {}
        result = _normalize_provider_section(section, "ollama")
        assert result["base_url"] == "http://localhost:11434/v1"
        assert result["default_model"] == "llama3"

    def test_custom_requires_base_url(self):
        section = {"api_key": "key"}
        with pytest.raises(ConfigError, match="base_url"):
            _normalize_provider_section(section, "custom.test-llm")

    def test_custom_with_base_url_passes(self):
        section = {"base_url": "http://localhost:8000/v1", "api_key": "key"}
        result = _normalize_provider_section(section, "custom.test-llm")
        assert result["base_url"] == "http://localhost:8000/v1"
        assert result["default_model"] == ""


class TestLoadProviderConfig:
    """load_provider_config tests."""

    def test_load_builtin(self):
        config = {
            "providers": {
                "default": "openai",
                "openai": {"api_key": "sk-test"},
            }
        }
        result = load_provider_config(config, "openai")
        assert result["api_key"] == "sk-test"
        assert result["base_url"] == "https://api.openai.com/v1"
        assert result["default_model"] == "gpt-4o"

    def test_load_custom(self):
        config = {
            "providers": {
                "default": "custom.my-llm",
                "custom": {
                    "my-llm": {
                        "api_key": "k",
                        "base_url": "http://localhost:8000/v1",
                        "default_model": "m",
                    },
                },
            }
        }
        result = load_provider_config(config, "custom.my-llm")
        assert result["api_key"] == "k"
        assert result["base_url"] == "http://localhost:8000/v1"
        assert result["default_model"] == "m"
