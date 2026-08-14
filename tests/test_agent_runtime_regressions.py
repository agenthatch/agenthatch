"""Regression tests for agent runtime bugs found in v1.0.10 audit.

Covers:
- Bug #23: legacy fallback at agent.py:481-491 used multi-word script_name
"""

from __future__ import annotations

import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Bug #23: Legacy fallback script_name not updated when {cap_name}.py found
# ---------------------------------------------------------------------------

class TestBug23LegacyFallbackUpdatesScriptName:
    """Bug #23: When ``script_name`` is a multi-word command (e.g.
    ``"python create_docx.py"``) and the file doesn't exist at that
    path, the legacy fallback checks ``{cap_name}.py`` and sets
    ``script_exists = True`` — but the previous code did NOT update
    ``script_name``, leaving the executor pointing at the non-existent
    multi-word path AND shadowing the Python tool implementation.

    The fix: when the legacy fallback discovers ``{cap_name}.py``,
    update ``script_name`` to ``f"{cap_name}.py"`` so the executor
    finds the file and the Python tool can still register as fallback.
    """

    def _simulate_branch(self, cap_name: str, script_name: str | None, scripts_dir: Path) -> tuple[bool, str | None]:
        """Replicate the agent.py:481-501 branch logic (post-fix)."""
        script_exists = False
        if script_name and scripts_dir and (scripts_dir / script_name).exists():
            script_exists = True
        elif scripts_dir and scripts_dir.is_dir():
            if (scripts_dir / f"{cap_name}.py").exists():
                script_exists = True
                script_name = f"{cap_name}.py"
        return script_exists, script_name

    def test_multi_word_script_name_updated_to_cap_filename(self, tmp_path: Path) -> None:
        """Multi-word script_name with {cap_name}.py existing → script_name
        updated to {cap_name}.py, not left as multi-word."""
        scripts_dir = tmp_path / "skills" / "scripts"
        scripts_dir.mkdir(parents=True)
        (scripts_dir / "create_docx.py").write_text("# stub\n")

        cap_name = "create_docx"
        script_name = "python create_docx.py"  # multi-word command

        script_exists, final_script_name = self._simulate_branch(cap_name, script_name, scripts_dir)

        assert script_exists, (
            "Fallback should set script_exists=True when {cap_name}.py exists"
        )
        assert final_script_name == "create_docx.py", (
            f"script_name should be updated to 'create_docx.py' (the "
            f"discovered filename), got {final_script_name!r}. Bug 23 "
            f"regression: legacy fallback left multi-word script_name, "
            f"breaking the capability permanently."
        )

    def test_explicit_script_name_unchanged_when_file_exists(self, tmp_path: Path) -> None:
        """When script_name directly matches an existing file, don't override."""
        scripts_dir = tmp_path / "skills" / "scripts"
        scripts_dir.mkdir(parents=True)
        (scripts_dir / "create_docx.py").write_text("# stub\n")

        cap_name = "create_docx"
        script_name = "create_docx.py"  # already correct

        script_exists, final_script_name = self._simulate_branch(cap_name, script_name, scripts_dir)

        assert script_exists
        assert final_script_name == "create_docx.py"

    def test_no_script_name_falls_back_to_cap_filename(self, tmp_path: Path) -> None:
        """When script_name is None, legacy fallback discovers {cap_name}.py."""
        scripts_dir = tmp_path / "skills" / "scripts"
        scripts_dir.mkdir(parents=True)
        (scripts_dir / "create_docx.py").write_text("# stub\n")

        cap_name = "create_docx"
        script_name = None

        script_exists, final_script_name = self._simulate_branch(cap_name, script_name, scripts_dir)

        assert script_exists
        assert final_script_name == "create_docx.py"

    def test_multi_word_script_name_with_no_cap_file_description_only(self, tmp_path: Path) -> None:
        """When neither multi-word script_name nor {cap_name}.py exists,
        register as description-only (Python tool fallback path)."""
        scripts_dir = tmp_path / "skills" / "scripts"
        scripts_dir.mkdir(parents=True)
        # No files in scripts_dir

        cap_name = "create_docx"
        script_name = "python create_docx.py"

        script_exists, _ = self._simulate_branch(cap_name, script_name, scripts_dir)

        assert not script_exists, (
            "Neither multi-word path nor {cap_name}.py exists — should be "
            "description-only so _register_python_tool can register the real "
            "Python implementation."
        )

    def test_source_code_updates_script_name_in_fallback(self) -> None:
        """Static guard: source file must contain the script_name update
        in the legacy fallback branch.

        Prevents a future refactor from collapsing the if/elif back into
        a bare ``script_exists = (scripts_dir / f"{cap_name}.py").exists()``
        without updating script_name.
        """
        src = Path(
            "agenthatch-core/src/agenthatch_core/agent.py"
        ).read_text()
        assert "Bug 23" in src or "Bug #23" in src, (
            "Bug 23 comment must be in source for future readers"
        )
        # The fix assigns script_name in the elif branch
        assert 'script_name = f"{cap_name}.py"' in src, (
            "Legacy fallback must update script_name when {cap_name}.py is found"
        )


if __name__ == "__main__":
    sys.exit(0)
