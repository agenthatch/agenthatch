"""Test agenthatch providers module (v0.2)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from agenthatch.exceptions import ProviderNotFoundError
from agenthatch.providers import (
    BUILTIN_PROVIDERS,
    get_default_provider,
    get_provider,
    list_builtin_providers,
    list_custom_providers,
    resolve_api_key,
    verify_api_key,
)


class TestBuiltinProviders:
    """Built-in provider registry tests."""

    def test_four_builtin_providers(self):
        assert len(BUILTIN_PROVIDERS) == 4
        assert "openai" in BUILTIN_PROVIDERS
        assert "anthropic" in BUILTIN_PROVIDERS
        assert "deepseek" in BUILTIN_PROVIDERS
        assert "ollama" in BUILTIN_PROVIDERS

    def test_openai_info(self):
        info = BUILTIN_PROVIDERS["openai"]
        assert info.name == "openai"
        assert info.kind == "builtin"
        assert info.env_key == "OPENAI_API_KEY"
        assert "api.openai.com" in info.base_url
        assert info.default_model == "gpt-4o"

    def test_ollama_no_env_key(self):
        info = BUILTIN_PROVIDERS["ollama"]
        assert info.env_key == ""

    def test_provider_info_is_frozen(self):
        info = BUILTIN_PROVIDERS["openai"]
        with pytest.raises(FrozenInstanceError):
            info.name = "changed"  # type: ignore[misc]


class TestGetProvider:
    """get_provider resolution tests."""

    def test_returns_builtin(self):
        info = get_provider("openai")
        assert info.name == "openai"
        assert info.kind == "builtin"

    def test_returns_custom_from_config(self):
        info = get_provider("custom.my-llm", {"providers": {"custom": {"my-llm": {
            "api_key": "test",
            "base_url": "http://localhost:8000/v1",
            "default_model": "llama",
        }}}})
        assert info.name == "custom.my-llm"
        assert info.kind == "custom"
        assert info.base_url == "http://localhost:8000/v1"

    def test_unknown_provider_raises(self):
        with pytest.raises(ProviderNotFoundError):
            get_provider("nonexistent")

    def test_missing_custom_provider_raises(self):
        with pytest.raises(ProviderNotFoundError):
            get_provider("custom.missing", {"providers": {"custom": {}}})


class TestResolveApiKey:
    """API key resolution priority chain tests."""

    def test_provider_env_var_highest(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        key = resolve_api_key("openai", config={}, prompt=False)
        assert key == "sk-from-env"

    def test_generic_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("AGENTHATCH_API_KEY", "sk-from-generic")
        key = resolve_api_key("openai", config={}, prompt=False)
        assert key == "sk-from-generic"

    def test_config_key_fallback(self):
        config = {
            "providers": {
                "openai": {"api_key": "sk-from-config"},
            }
        }
        key = resolve_api_key("openai", config=config, prompt=False)
        assert key == "sk-from-config"

    def test_returns_none_when_no_key(self):
        key = resolve_api_key("openai", config={}, prompt=False)
        assert key is None

    def test_no_prompt_skips_interactive(self, monkeypatch):
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        key = resolve_api_key("openai", config={}, prompt=False)
        assert key is None

    def test_custom_provider_config_key(self):
        config = {
            "providers": {
                "custom": {
                    "my-llm": {"api_key": "sk-custom"},
                }
            }
        }
        key = resolve_api_key("custom.my-llm", config=config, prompt=False)
        assert key == "sk-custom"


class TestVerifyApiKey:
    """API key connectivity verification tests."""

    def test_success(self, mock_httpx_success):
        ok, detail = verify_api_key("openai", "sk-test", "https://api.openai.com/v1")
        assert ok is True
        assert "200" in detail

    def test_unauthorized(self, mock_httpx_unauthorized):
        ok, detail = verify_api_key("openai", "sk-bad", "https://api.openai.com/v1")
        assert ok is False
        assert "401" in detail

    def test_timeout_returns_uncertain(self, monkeypatch):
        import httpx

        def _mock_timeout(*args, **kwargs):
            raise httpx.TimeoutException("timed out")

        monkeypatch.setattr("agenthatch.providers.httpx.get", _mock_timeout)
        ok, detail = verify_api_key("openai", "sk-test", "https://api.openai.com/v1")
        assert ok is True  # uncertain, not failure
        assert "timed out" in detail


class TestListProviders:
    """Provider listing tests."""

    def test_list_builtin(self):
        providers = list_builtin_providers()
        assert len(providers) == 4

    def test_list_custom_empty(self):
        providers = list_custom_providers({})
        assert len(providers) == 0

    def test_list_custom_with_entries(self):
        config = {
            "providers": {
                "custom": {
                    "a": {"base_url": "http://a.com", "default_model": "m1"},
                    "b": {"base_url": "http://b.com", "default_model": "m2"},
                }
            }
        }
        providers = list_custom_providers(config)
        assert len(providers) == 2
        names = {p.name for p in providers}
        assert "custom.a" in names
        assert "custom.b" in names


class TestGetDefaultProvider:
    """get_default_provider tests."""

    def test_default(self):
        assert get_default_provider({}) == "openai"

    def test_from_config(self):
        config = {"providers": {"default": "anthropic"}}
        assert get_default_provider(config) == "anthropic"
