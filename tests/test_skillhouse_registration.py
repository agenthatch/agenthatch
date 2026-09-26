"""Regression: skillhouse registration must record the SOURCE dir.

`hatch <name> -o <dir>` used to register the output dir's
agenthatch.yaml as ``ahs_path``. By-name resolution then treated
``ahs_path.parent`` as the skill source dir — the output dir has no
SKILL.md, so ``hatch <name>`` crashed with "No .md file found".
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

from agenthatch.cli.commands.hatch import _register_skillhouse


def _fake_ahspec() -> Any:
    """Minimal stand-in for the AHSSpec attributes add_entry reads."""
    return NS(
        identity=NS(id="my-skill", display_name="My Skill", version="1"),
        intent=NS(triggers=[], satisfies=[], summary=""),
        interface=NS(provides=[], requires=[], compatible_with=[]),
        agent=NS(status="hatched", hatched_at=""),
    )


class TestRegisterSkillhouseSourceDir:
    def test_ahs_path_always_points_at_skill_dir(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """Registered ahs_path is skill_dir/agenthatch.yaml even with -o."""
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        # The -o output dir exists too — registration must ignore it.
        output_dir = tmp_path / "elsewhere"
        output_dir.mkdir()

        captured: dict[str, Any] = {}

        class FakeIndex:
            entry_count = 1

            def __init__(self, path: str) -> None:
                pass

            def add_entry(
                self, skill_id: str, spec: Any, ahs_path: str
            ) -> None:
                captured["skill_id"] = skill_id
                captured["ahs_path"] = ahs_path

        monkeypatch.setattr(
            "agenthatch.house.index.SkillhouseIndex", FakeIndex
        )

        config = {"skillhouse": {"path": str(tmp_path / "skillhouse.json")}}
        _register_skillhouse(_fake_ahspec(), skill_dir, config)

        assert captured["skill_id"] == "my-skill"
        assert captured["ahs_path"] == str(skill_dir / "agenthatch.yaml")
        assert "elsewhere" not in captured["ahs_path"]

    def test_by_name_resolution_derives_skill_dir(
        self, tmp_path: Path
    ) -> None:
        """ahs_path.parent must be a dir containing SKILL.md.

        Documents the downstream contract that made the -o variant a
        crash: ``_resolve_from_index`` uses ahs_path.parent as skill dir.
        """
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# My Skill", encoding="utf-8")

        ahs_path = skill_dir / "agenthatch.yaml"

        assert (ahs_path.parent / "SKILL.md").is_file()
