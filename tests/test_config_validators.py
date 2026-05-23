"""Test config validators (v0.2)."""

from __future__ import annotations

import pytest

from agenthatch.config.validators import (
    validate_base_url,
    validate_config_integrity,
    validate_provider_name,
)
from agenthatch.exceptions import ConfigError


class TestValidateProviderName:
    """Provider name validation tests."""

    def test_valid_names(self):
        validate_provider_name("openai")
        validate_provider_name("my-llm")
        validate_provider_name("custom.provider")
        validate_provider_name("a1")

    def test_too_short(self):
        with pytest.raises(ConfigError):
            validate_provider_name("a")

    def test_starts_with_number(self):
        with pytest.raises(ConfigError):
            validate_provider_name("1provider")

    def test_special_chars(self):
        with pytest.raises(ConfigError):
            validate_provider_name("my provider")


class TestValidateBaseUrl:
    """Base URL validation tests."""

    def test_valid_urls(self):
        validate_base_url("https://api.openai.com/v1")
        validate_base_url("http://localhost:11434/v1")
        validate_base_url("https://api.deepseek.com")

    def test_invalid_url(self):
        with pytest.raises(ConfigError):
            validate_base_url("ftp://bad.com")

    def test_no_protocol(self):
        with pytest.raises(ConfigError):
            validate_base_url("api.openai.com")


class TestValidateConfigIntegrity:
    """Config integrity validation tests."""

    def test_valid_config(self):
        config = {
            "providers": {
                "default": "openai",
            }
        }
        warnings = validate_config_integrity(config)
        assert len(warnings) == 0

    def test_missing_providers_section(self):
        with pytest.raises(ConfigError):
            validate_config_integrity({})

    def test_unknown_default_provider(self):
        config = {
            "providers": {
                "default": "cohere",
            }
        }
        warnings = validate_config_integrity(config)
        assert len(warnings) > 0
        assert "cohere" in warnings[0]

    def test_custom_provider_missing_url(self):
        config = {
            "providers": {
                "default": "openai",
                "custom": {
                    "test-llm": {"default_model": "m1"},
                },
            }
        }
        warnings = validate_config_integrity(config)
        assert len(warnings) > 0
        assert "base_url" in warnings[0].lower()
