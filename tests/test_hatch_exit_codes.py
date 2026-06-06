"""Test hatch command exit codes (v0.5).

Verifies that the hatch command handles errors gracefully
by returning the correct exit codes for each error scenario.
"""

from __future__ import annotations


class TestHatchExitCode1:
    """Exit code 1: Skill not found / invalid path."""

    def test_nonexistent_path(self, runner, app, tmp_agenthatch_home):
        """Non-existent absolute path should exit with code 1."""
        result = runner.invoke(app, ["hatch", "/nonexistent/path/to/skill"])
        assert result.exit_code == 1

    def test_nonexistent_relative_path(self, runner, app, tmp_agenthatch_home):
        """Non-existent relative path should exit with code 1."""
        result = runner.invoke(app, ["hatch", "nonexistent-skill-dir"])
        assert result.exit_code == 1

    def test_directory_without_skill_md(self, runner, app, tmp_path, tmp_agenthatch_home):
        """Directory that exists but has no SKILL.md should exit with code 1."""
        empty_dir = tmp_path / "empty-dir"
        empty_dir.mkdir()
        result = runner.invoke(app, ["hatch", str(empty_dir)])
        assert result.exit_code == 1

    def test_file_without_skill_md(self, runner, app, tmp_path, tmp_agenthatch_home):
        """File that is not a SKILL.md should exit with code 1."""
        not_skill = tmp_path / "README.md"
        not_skill.write_text("# Not a skill")
        result = runner.invoke(app, ["hatch", str(not_skill)])
        assert result.exit_code == 1


class TestHatchExitCode2:
    """Exit code 2: agenthatch.yaml already exists without --force.

    Note: As of v0.6.1, existing agenthatch.yaml no longer prevents
    Phase 3 (Agent generation). The hatch command now skips yaml write
    but continues to generate the Agent directory. Exit code 2 is no
    longer returned for this scenario.
    """

    def test_existing_output_without_force(
        self, runner, app, tmp_path, tmp_agenthatch_home, monkeypatch
    ):
        """Existing agenthatch.yaml without --force should succeed and skip yaml write."""
        # Create a skill directory with SKILL.md
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: test\n---\n# Test Skill")

        # Create an existing agenthatch.yaml
        (skill_dir / "agenthatch.yaml").write_text("identity:\n  id: test\n")

        # Mock assemble_context to return a valid context
        from unittest.mock import MagicMock

        mock_ctx = MagicMock()
        mock_ctx.dir_name = "test-skill"
        mock_ctx.frontmatter = {"name": "test"}
        mock_ctx.file_manifest = MagicMock()
        mock_ctx.file_manifest.entries = []
        mock_ctx.file_manifest.content_bundle.return_value = []
        mock_ctx.parse_warnings = []

        monkeypatch.setattr(
            "agenthatch.cli.commands.hatch.assemble_context",
            lambda path: mock_ctx,
        )

        # Mock build_ahspec to succeed
        mock_spec = MagicMock()
        mock_spec.identity.id = "test"
        mock_spec.confidence_report = None
        mock_spec.model_dump_json.return_value = '{"identity": {"id": "test"}}'

        def _mock_build(context, config, **kwargs):
            return mock_spec, {}

        monkeypatch.setattr(
            "agenthatch.skill.builder.build_ahspec",
            _mock_build,
        )

        result = runner.invoke(app, ["hatch", str(skill_dir)])
        assert result.exit_code == 0
        assert "already exists" in result.output


class TestHatchExitCode3:
    """Exit code 3: Parse error from assemble_context."""

    def test_parse_error(self, runner, app, tmp_path, tmp_agenthatch_home, monkeypatch):
        """Parse error from assemble_context should exit with code 3."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: test\n---\n# Test")

        def _raise_parse_error(path):
            raise Exception("Parse error in SKILL.md")

        monkeypatch.setattr(
            "agenthatch.cli.commands.hatch.assemble_context",
            _raise_parse_error,
        )

        result = runner.invoke(app, ["hatch", str(skill_dir)])
        assert result.exit_code == 3
        assert "Parse error" in result.output


class TestHatchExitCode4:
    """Exit code 4: Inference error from build_ahspec."""

    def test_inference_error(self, runner, app, tmp_path, tmp_agenthatch_home, monkeypatch):
        """Inference error from build_ahspec should exit with code 4."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: test\n---\n# Test")

        from unittest.mock import MagicMock

        mock_ctx = MagicMock()
        mock_ctx.dir_name = "test-skill"
        mock_ctx.frontmatter = {"name": "test"}
        mock_ctx.file_manifest = MagicMock()
        mock_ctx.file_manifest.entries = []
        mock_ctx.file_manifest.content_bundle.return_value = []
        mock_ctx.parse_warnings = []

        monkeypatch.setattr(
            "agenthatch.cli.commands.hatch.assemble_context",
            lambda path: mock_ctx,
        )

        def _raise_inference_error(context, config, **kwargs):
            raise Exception("LLM inference failed")

        monkeypatch.setattr(
            "agenthatch.skill.builder.build_ahspec",
            _raise_inference_error,
        )

        result = runner.invoke(app, ["hatch", str(skill_dir)])
        assert result.exit_code == 4
        assert "Inference error" in result.output
