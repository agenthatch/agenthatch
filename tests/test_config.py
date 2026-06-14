"""Test Config class (v0.2)."""

from __future__ import annotations

import pytest

from agenthatch.config import Config
from agenthatch.exceptions import ConfigError


class TestConfigLoad:
    """Config.load() tests."""

    def test_load_empty_when_no_file(self, tmp_path):
        result = Config.load(config_path=tmp_path / "nonexistent.toml")
        assert isinstance(result, dict)

    def test_load_valid_toml(self, tmp_path):
        path = tmp_path / "test.toml"
        path.write_text('[core]\nverbose = true\n\n[providers]\ndefault = "openai"\n')
        result = Config.load(config_path=path)
        assert result["core"]["verbose"] is True
        assert result["providers"]["default"] == "openai"

    def test_load_invalid_toml_raises(self, tmp_path):
        path = tmp_path / "bad.toml"
        path.write_text("not valid toml [[[")
        with pytest.raises(ConfigError, match="Failed to read"):
            Config.load(config_path=path)


class TestConfigApplyEnvOverrides:
    """Config._apply_env_overrides() tests."""

    def test_verbose_env(self, monkeypatch):
        monkeypatch.setenv("AGENTHATCH_VERBOSE", "true")
        result = Config._apply_env_overrides({})
        assert result["core"]["verbose"] is True

    def test_verbose_env_false(self, monkeypatch):
        monkeypatch.setenv("AGENTHATCH_VERBOSE", "0")
        result = Config._apply_env_overrides({})
        assert result["core"]["verbose"] is False

    def test_provider_env(self, monkeypatch):
        monkeypatch.setenv("AGENTHATCH_PROVIDER", "anthropic")
        result = Config._apply_env_overrides({})
        # v0.9: provider default moved from [providers].default to [agenthatch].default
        assert result["agenthatch"]["default"] == "anthropic"

    def test_llm_provider_env(self, monkeypatch):
        monkeypatch.setenv("AGENTHATCH_LLM_PROVIDER", "deepseek")
        result = Config._apply_env_overrides({})
        assert result["agenthatch"]["default"] == "deepseek"

    def test_provider_env_priority(self, monkeypatch):
        monkeypatch.setenv("AGENTHATCH_PROVIDER", "openai")
        monkeypatch.setenv("AGENTHATCH_LLM_PROVIDER", "anthropic")
        result = Config._apply_env_overrides({})
        # AGENTHATCH_PROVIDER is processed first, AGENTHATCH_LLM_PROVIDER overwrites
        assert result["agenthatch"]["default"] == "anthropic"

    def test_no_env_returns_unchanged(self):
        result = Config._apply_env_overrides({"core": {"verbose": True}})
        assert result["core"]["verbose"] is True


class TestConfigCreateDefault:
    """Config.create_default() tests."""

    def _mock_config_paths(self, monkeypatch, tmp_path):
        """Helper to mock CONFIG_DIR and CONFIG_FILE."""
        config_dir = tmp_path / ".agenthatch"
        config_file = config_dir / "config.toml"
        monkeypatch.setattr("agenthatch.config.CONFIG_DIR", config_dir)
        monkeypatch.setattr("agenthatch.config.CONFIG_FILE", config_file)
        return config_file

    def test_creates_config_file(self, monkeypatch, tmp_path):
        self._mock_config_paths(monkeypatch, tmp_path)
        path = Config.create_default()
        assert path.exists()
        content = path.read_text()
        assert "[core]" in content
        assert "[providers]" in content
        assert 'default = "openai"' in content

    def test_force_overwrite(self, monkeypatch, tmp_path):
        self._mock_config_paths(monkeypatch, tmp_path)
        Config.create_default()
        path = Config.create_default(force=True)
        assert path.exists()

    def test_refuses_overwrite(self, monkeypatch, tmp_path):
        self._mock_config_paths(monkeypatch, tmp_path)
        Config.create_default()
        with pytest.raises(ConfigError, match="already exists"):
            Config.create_default()

    def test_custom_provider(self, monkeypatch, tmp_path):
        self._mock_config_paths(monkeypatch, tmp_path)
        path = Config.create_default(provider="anthropic")
        content = path.read_text()
        assert 'default = "anthropic"' in content
