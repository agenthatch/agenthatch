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

    def test_builtin_providers_present(self):
        assert len(BUILTIN_PROVIDERS) >= 6
        assert "openai" in BUILTIN_PROVIDERS
        assert "anthropic" in BUILTIN_PROVIDERS
        assert "deepseek" in BUILTIN_PROVIDERS
        assert "ollama" in BUILTIN_PROVIDERS
        assert "glm" in BUILTIN_PROVIDERS
        assert "qwen" in BUILTIN_PROVIDERS

    def test_openai_info(self):
        info = BUILTIN_PROVIDERS["openai"]
        assert info.name == "openai"
        assert info.kind == "builtin"
        assert info.env_key == "OPENAI_API_KEY"
        assert "api.openai.com" in info.base_url
        assert info.default_model == "gpt-5.6-sol"

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

    def test_timeout_returns_failure(self, monkeypatch):
        import httpx

        def _mock_timeout(*args, **kwargs):
            raise httpx.TimeoutException("timed out")

        monkeypatch.setattr("agenthatch.providers.httpx.get", _mock_timeout)
        ok, detail = verify_api_key("openai", "sk-test", "https://api.openai.com/v1")
        assert ok is False  # M1 fix: timeout → failure, not "uncertain"
        assert "timed out" in detail

    def test_glm_v4_base_url_hits_v4_models(self, monkeypatch):
        """GLM base_url ends with /v4 — models URL must be /v4/models."""
        captured: dict[str, str] = {}

        class _MockResponse:
            status_code = 200
            is_success = True

        def _mock_get(url, **kwargs):
            captured["url"] = url
            return _MockResponse()

        monkeypatch.setattr("agenthatch.providers.httpx.get", _mock_get)
        ok, _ = verify_api_key(
            "glm", "sk-test", "https://open.bigmodel.cn/api/paas/v4"
        )
        assert ok is True
        assert captured["url"] == "https://open.bigmodel.cn/api/paas/v4/models"

    def test_v1_base_url_appends_models_only(self, monkeypatch):
        """base_url ending in /v1 must keep appending just /models."""
        captured: dict[str, str] = {}

        class _MockResponse:
            status_code = 200
            is_success = True

        def _mock_get(url, **kwargs):
            captured["url"] = url
            return _MockResponse()

        monkeypatch.setattr("agenthatch.providers.httpx.get", _mock_get)
        ok, _ = verify_api_key(
            "qwen", "sk-test", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        assert ok is True
        assert captured["url"] == (
            "https://dashscope.aliyuncs.com/compatible-mode/v1/models"
        )

    def test_unversioned_base_url_appends_v1_models(self, monkeypatch):
        """base_url without a version segment keeps the /v1/models convention."""
        captured: dict[str, str] = {}

        class _MockResponse:
            status_code = 200
            is_success = True

        def _mock_get(url, **kwargs):
            captured["url"] = url
            return _MockResponse()

        monkeypatch.setattr("agenthatch.providers.httpx.get", _mock_get)
        ok, _ = verify_api_key(
            "anthropic", "sk-test", "https://api.anthropic.com"
        )
        assert ok is True
        assert captured["url"] == "https://api.anthropic.com/v1/models"


class TestGlmQwenPresets:
    """GLM / Qwen builtin provider preset tests."""

    def test_glm_preset(self):
        info = BUILTIN_PROVIDERS["glm"]
        assert info.name == "glm"
        assert info.kind == "builtin"
        assert info.env_key == "ZAI_API_KEY"
        assert info.base_url == "https://open.bigmodel.cn/api/paas/v4"
        assert info.default_model == "glm-5"
        assert info.context_window == 200000
        assert info.features.supports_tools
        assert info.features.supports_stream_tools
        assert info.features.supports_reasoning_content
        assert "glm-5" in info.features.available_models

    def test_qwen_preset(self):
        info = BUILTIN_PROVIDERS["qwen"]
        assert info.name == "qwen"
        assert info.kind == "builtin"
        assert info.env_key == "DASHSCOPE_API_KEY"
        assert info.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
        assert info.default_model == "qwen3.8-max"
        assert info.context_window == 262144
        assert info.features.supports_tools
        assert info.features.supports_stream_tools
        assert info.features.supports_reasoning_content
        assert "qwen3.8-max" in info.features.available_models

    def test_get_provider_resolves_glm_qwen(self):
        for name in ("glm", "qwen"):
            info = get_provider(name)
            assert info.kind == "builtin"
            assert info.name == name

    def test_ollama_default_model_updated(self):
        info = BUILTIN_PROVIDERS["ollama"]
        assert info.default_model == "llama3.1"


class TestDeepSeekBaseURL:
    """C1 fix: deepseek base_url must include /v1 for OpenAI client compat."""

    def test_base_url_has_v1_suffix(self):
        from agenthatch.providers import BUILTIN_PROVIDERS
        deepseek = BUILTIN_PROVIDERS["deepseek"]
        assert deepseek.base_url == "https://api.deepseek.com/v1", (
            "C1 regression: deepseek base_url must include /v1 suffix "
            "for OpenAI client compatibility"
        )

    def test_openai_base_url_has_v1_suffix(self):
        from agenthatch.providers import BUILTIN_PROVIDERS
        openai_p = BUILTIN_PROVIDERS["openai"]
        assert openai_p.base_url == "https://api.openai.com/v1"

    def test_ollama_base_url_has_v1_suffix(self):
        from agenthatch.providers import BUILTIN_PROVIDERS
        ollama = BUILTIN_PROVIDERS["ollama"]
        assert ollama.base_url == "http://localhost:11434/v1"


class TestListProviders:
    """Provider listing tests."""

    def test_list_builtin(self):
        providers = list_builtin_providers()
        assert len(providers) >= 4

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
        # v0.9: default provider moved from [providers].default to [agenthatch].default
        config = {"agenthatch": {"default": "anthropic"}}
        assert get_default_provider(config) == "anthropic"
