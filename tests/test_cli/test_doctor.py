"""Test agenthatch doctor command (v0.2)."""



class TestDoctor:
    """agenthatch doctor tests."""

    def test_basic_run_no_config(self, runner, app, tmp_agenthatch_home):
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code in (0, 1)
        assert "Python" in result.output
        assert "No config file found" in result.output

    def test_api_key_configured(
        self, runner, app, tmp_agenthatch_home, mock_httpx_success, monkeypatch
    ):
        _write_minimal_config(tmp_agenthatch_home)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setattr(
            "agenthatch.cli.commands.doctor.CONFIG_FILE",
            tmp_agenthatch_home / "config.toml",
        )
        result = runner.invoke(app, ["doctor"])
        assert "Provider:" in result.output

    def test_api_key_missing(
        self, runner, app, tmp_agenthatch_home, monkeypatch
    ):
        _write_minimal_config(tmp_agenthatch_home, provider="openai")
        monkeypatch.setattr(
            "agenthatch.cli.commands.doctor.CONFIG_FILE",
            tmp_agenthatch_home / "config.toml",
        )
        result = runner.invoke(app, ["doctor"])
        assert "not configured" in result.output or result.exit_code == 1

    def test_api_key_bad(
        self, runner, app, tmp_agenthatch_home, mock_httpx_unauthorized, monkeypatch
    ):
        _write_minimal_config(tmp_agenthatch_home)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-bad-key")
        monkeypatch.setattr(
            "agenthatch.cli.commands.doctor.CONFIG_FILE",
            tmp_agenthatch_home / "config.toml",
        )
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 1
        assert "failed" in result.output.lower() or "FAIL" in result.output

    def test_ollama_skips_key_check(
        self, runner, app, tmp_agenthatch_home, monkeypatch
    ):
        _write_minimal_config(tmp_agenthatch_home, provider="ollama")
        monkeypatch.setattr(
            "agenthatch.cli.commands.doctor.CONFIG_FILE",
            tmp_agenthatch_home / "config.toml",
        )
        result = runner.invoke(app, ["doctor"])
        assert "no key needed" in result.output or result.exit_code == 0


def _write_minimal_config(tmp_path, provider: str = "openai"):
    """Write a minimal config.toml for doctor tests."""
    config_content = f"""\
[core]
verbose = false

[providers]
default = "{provider}"

[providers.openai]
api_key = ""
base_url = "https://api.openai.com/v1"
default_model = "gpt-4o"

[providers.anthropic]
api_key = ""
base_url = "https://api.anthropic.com"
default_model = "claude-sonnet-4-20250514"

[providers.deepseek]
api_key = ""
base_url = "https://api.deepseek.com"
default_model = "deepseek-chat"

[providers.ollama]
api_key = ""
base_url = "http://localhost:11434/v1"
default_model = "llama3"
"""
    tmp_path.joinpath("config.toml").write_text(config_content)
