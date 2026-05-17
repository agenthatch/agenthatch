"""Test agenthatch init command."""

from unittest.mock import patch


class TestInit:
    """agenthatch init tests."""

    def test_creates_config(self, runner, app, tmp_agenthatch_home):
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0
        assert "Config created" in result.output
        assert tmp_agenthatch_home.joinpath("config.toml").exists()

    def test_config_content(self, runner, app, tmp_agenthatch_home):
        runner.invoke(app, ["init"])
        content = tmp_agenthatch_home.joinpath("config.toml").read_text()
        assert "[core]" in content
        assert "[providers]" in content
        assert "OPENAI_API_KEY" in content

    def test_refuses_overwrite_without_force(self, runner, app, tmp_agenthatch_home):
        tmp_agenthatch_home.joinpath("config.toml").write_text("# existing")
        with patch("rich.prompt.Confirm.ask", return_value=False):
            result = runner.invoke(app, ["init"])
        assert result.exit_code == 2
        assert "already exists" in result.output

    def test_force_overwrite(self, runner, app, tmp_agenthatch_home):
        tmp_agenthatch_home.joinpath("config.toml").write_text("# old")
        result = runner.invoke(app, ["init", "--force"])
        assert result.exit_code == 0
        content = tmp_agenthatch_home.joinpath("config.toml").read_text()
        assert "agenthatch configuration file" in content

    def test_confirm_overwrite(self, runner, app, tmp_agenthatch_home):
        tmp_agenthatch_home.joinpath("config.toml").write_text("# old")
        with patch("rich.prompt.Confirm.ask", return_value=True):
            result = runner.invoke(app, ["init"])
        assert result.exit_code == 0
        content = tmp_agenthatch_home.joinpath("config.toml").read_text()
        assert "agenthatch configuration file" in content
