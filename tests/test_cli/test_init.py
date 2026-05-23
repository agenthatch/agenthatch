"""Test agenthatch init command (v0.2)."""



class TestInitInteractive:
    """Interactive init flow tests."""

    def test_select_openai(self, runner, app, tmp_agenthatch_home):
        result = runner.invoke(
            app,
            ["init"],
            input="1\nsk-test-key\ngpt-4o\n",
        )
        assert result.exit_code == 0
        content = tmp_agenthatch_home.joinpath("config.toml").read_text()
        assert 'default = "openai"' in content

    def test_select_anthropic(self, runner, app, tmp_agenthatch_home):
        result = runner.invoke(
            app,
            ["init"],
            input="2\nsk-ant-test-key\nclaude-sonnet-4-20250514\n",
        )
        assert result.exit_code == 0
        content = tmp_agenthatch_home.joinpath("config.toml").read_text()
        assert 'default = "anthropic"' in content

    def test_select_deepseek(self, runner, app, tmp_agenthatch_home):
        result = runner.invoke(
            app,
            ["init"],
            input="3\nsk-deepseek-key\ndeepseek-chat\n",
        )
        assert result.exit_code == 0
        content = tmp_agenthatch_home.joinpath("config.toml").read_text()
        assert 'default = "deepseek"' in content

    def test_select_ollama_no_key(self, runner, app, tmp_agenthatch_home):
        result = runner.invoke(
            app,
            ["init"],
            input="4\n\nllama3\n",
        )
        assert result.exit_code == 0
        content = tmp_agenthatch_home.joinpath("config.toml").read_text()
        assert 'default = "ollama"' in content

    def test_refuses_overwrite_without_force(
        self, runner, app, tmp_agenthatch_home
    ):
        tmp_agenthatch_home.joinpath("config.toml").write_text("# existing")
        result = runner.invoke(app, ["init"], input="n\n")
        assert result.exit_code == 2
        assert "already exists" in result.output

    def test_force_overwrite(self, runner, app, tmp_agenthatch_home):
        tmp_agenthatch_home.joinpath("config.toml").write_text("# old config")
        result = runner.invoke(
            app,
            ["init", "--force"],
            input="1\nsk-overwrite-key\ngpt-4o\n",
        )
        assert result.exit_code == 0
        content = tmp_agenthatch_home.joinpath("config.toml").read_text()
        assert 'default = "openai"' in content

    def test_custom_provider(self, runner, app, tmp_agenthatch_home):
        result = runner.invoke(
            app,
            ["init"],
            input="5\nmy-llm\nhttp://localhost:8000/v1\nmy-key\nmixtral-8x7b\nMY_LLM_KEY\n",
        )
        assert result.exit_code == 0
        content = tmp_agenthatch_home.joinpath("config.toml").read_text()
        assert "custom.my-llm" in content
        assert "http://localhost:8000/v1" in content


class TestInitProviderFlag:
    """Test --provider flag for init."""

    def test_provider_flag_openai(self, runner, app, tmp_agenthatch_home):
        result = runner.invoke(
            app,
            ["init", "--provider", "openai"],
            input="sk-flag-key\ngpt-4o-mini\n",
        )
        assert result.exit_code == 0

    def test_provider_flag_unknown(self, runner, app, tmp_agenthatch_home):
        result = runner.invoke(
            app,
            ["init", "--provider", "nonexistent"],
        )
        assert result.exit_code == 2


class TestInitNonInteractive:
    """Test --non-interactive init mode."""

    def test_non_interactive_openai(
        self, runner, app, tmp_agenthatch_home, monkeypatch
    ):
        monkeypatch.setenv("AGENTHATCH_PROVIDER", "openai")
        monkeypatch.setenv("AGENTHATCH_LLM_MODEL", "gpt-4o")
        result = runner.invoke(app, ["init", "--non-interactive"])
        assert result.exit_code == 0
        content = tmp_agenthatch_home.joinpath("config.toml").read_text()
        assert 'default = "openai"' in content

    def test_non_interactive_anthropic(
        self, runner, app, tmp_agenthatch_home, monkeypatch
    ):
        monkeypatch.setenv("AGENTHATCH_PROVIDER", "anthropic")
        result = runner.invoke(app, ["init", "--non-interactive"])
        assert result.exit_code == 0
        content = tmp_agenthatch_home.joinpath("config.toml").read_text()
        assert 'default = "anthropic"' in content


class TestInitConfigContent:
    """Validate the written config.toml content."""

    def test_config_has_providers(self, runner, app, tmp_agenthatch_home):
        result = runner.invoke(
            app,
            ["init", "--force"],
            input="1\nsk-test\ngpt-4o\n",
        )
        assert result.exit_code == 0
        content = tmp_agenthatch_home.joinpath("config.toml").read_text()
        assert "[core]" in content
        assert "[providers]" in content
        assert "[providers.openai]" in content
        assert "[providers.anthropic]" in content
        assert "[providers.deepseek]" in content
        assert "[providers.ollama]" in content
        assert 'default = "openai"' in content
