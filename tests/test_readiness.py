"""Tests for the readiness environment audit (generate/readiness.py).

Regression coverage for the false "missing system tool" warnings:
``base.dependencies`` may list Python packages (pypdf, ...) alongside
real CLI tools — an installed Python package is never on PATH.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from agenthatch.generate.readiness import (
    DependencyManifest,
    _is_python_package_available,
    audit_environment,
    extract_dependencies,
)


class TestScriptDependencyDetection:
    def test_detects_imports_from_scripts_dir(self, tmp_path: Path) -> None:
        """Conventional layout <skill>/scripts is scanned."""
        skill_dir = tmp_path / "my-skill"
        (skill_dir / "scripts").mkdir(parents=True)
        (skill_dir / "scripts" / "run.py").write_text(
            "import pypdf\nimport json\n", encoding="utf-8"
        )

        deps = extract_dependencies(skill_dir, {})

        assert "pypdf" in deps.pip_packages
        assert "json" not in deps.pip_packages  # stdlib is filtered out

    def test_detects_imports_from_skills_scripts_dir(
        self, tmp_path: Path
    ) -> None:
        """Generated-agent layout <skill>/skills/scripts also works."""
        skill_dir = tmp_path / "my-skill"
        (skill_dir / "skills" / "scripts").mkdir(parents=True)
        (skill_dir / "skills" / "scripts" / "run.py").write_text(
            "import pdfplumber\n", encoding="utf-8"
        )

        deps = extract_dependencies(skill_dir, {})

        assert "pdfplumber" in deps.pip_packages


class TestPythonPackageFallback:
    def test_stdlib_module_is_available(self) -> None:
        assert _is_python_package_available("json") is True

    def test_unknown_name_is_not_available(self) -> None:
        assert (
            _is_python_package_available("definitely-not-a-real-pkg-xyz")
            is False
        )

    def test_installed_package_not_reported_as_missing_system_tool(
        self,
    ) -> None:
        """A pip package in base.dependencies must not fail the PATH check."""
        manifest = DependencyManifest(system_tools=["json"])

        report = audit_environment(manifest)

        assert report.system_tools["json"] is True

    def test_real_cli_tool_still_found_on_path(self) -> None:
        """The plain PATH check is untouched for genuine executables."""
        tool = next(
            (t for t in ("python3", "python", "pip") if shutil.which(t)),
            None,
        )
        if tool is None:
            pytest.skip("no CLI tool on PATH")

        report = audit_environment(DependencyManifest(system_tools=[tool]))

        assert report.system_tools[tool] is True
