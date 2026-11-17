"""Regression tests for deterministic Python script signature extraction."""

from pathlib import Path

import pytest

from agenthatch.skill.parser import (
    ToolSchema,
    analyze_scripts,
    analyze_scripts_from_manifest,
    extract_python_signatures,
)
from agenthatch.skill.spec import FileEntry, FileManifest

PYTHON_SCRIPT = '''\
def public_sync(name: str, count: int) -> bool:
    """Return whether the named count is valid."""
    return bool(name and count)


async def public_async(value: float) -> dict[str, int]:
    """Return an asynchronously computed result."""
    return {"value": int(value)}


def _private(hidden: str) -> None:
    return None


class PublicClass:
    def class_method(self, value: str) -> str:
        return value


def outer() -> None:
    def nested(inner: int) -> int:
        return inner

    nested(1)


def unannotated(value):
    return value
'''


def _assert_signatures(functions: list[ToolSchema], source_file: str) -> None:
    assert [function.name for function in functions] == [
        "public_sync",
        "public_async",
        "outer",
        "unannotated",
    ]
    assert functions[0].args == [
        {"name": "name", "type": "str"},
        {"name": "count", "type": "int"},
    ]
    assert functions[0].returns == "bool"
    assert functions[0].docstring == "Return whether the named count is valid."
    assert functions[0].source_file == source_file
    assert functions[1].args == [{"name": "value", "type": "float"}]
    assert functions[1].returns == "dict[str, int]"
    assert functions[1].docstring == "Return an asynchronously computed result."
    assert functions[1].source_file == source_file
    assert functions[2].args == []
    assert functions[2].returns == "None"
    assert functions[2].docstring is None
    assert functions[2].source_file == source_file
    assert functions[3].args == [{"name": "value", "type": None}]
    assert functions[3].returns is None
    assert functions[3].docstring is None
    assert functions[3].source_file == source_file


def test_extract_python_signatures_file(tmp_path: Path) -> None:
    """The file entry point extracts public sync and async functions only."""
    script = tmp_path / "api.py"
    script.write_text(PYTHON_SCRIPT, encoding="utf-8")

    _assert_signatures(extract_python_signatures(script), str(script))


def test_analyze_scripts_from_skill_directory(tmp_path: Path) -> None:
    """The skill-directory entry point preserves source file paths."""
    script = tmp_path / "skills" / "scripts" / "api.py"
    script.parent.mkdir(parents=True)
    script.write_text(PYTHON_SCRIPT, encoding="utf-8")

    manifest = analyze_scripts(tmp_path)

    _assert_signatures(manifest.python_functions, str(script))


def test_analyze_scripts_from_manifest() -> None:
    """The in-memory manifest entry point extracts the same signatures."""
    file_manifest = FileManifest(entries=[
        FileEntry(
            path="skills/scripts/api.py",
            hash="",
            size_bytes=len(PYTHON_SCRIPT),
            content=PYTHON_SCRIPT,
        )
    ])

    manifest = analyze_scripts_from_manifest(file_manifest)

    _assert_signatures(manifest.python_functions, "skills/scripts/api.py")


@pytest.mark.parametrize("entrypoint", ["file", "manifest"])
def test_invalid_python_returns_empty(tmp_path: Path, entrypoint: str) -> None:
    """Invalid Python is ignored by both extraction entry points."""
    invalid = "def broken(:\n    pass\n"
    if entrypoint == "file":
        script = tmp_path / "broken.py"
        script.write_text(invalid, encoding="utf-8")
        assert extract_python_signatures(script) == []
    else:
        file_manifest = FileManifest(entries=[
            FileEntry(
                path="broken.py",
                hash="",
                size_bytes=len(invalid),
                content=invalid,
            )
        ])
        assert analyze_scripts_from_manifest(file_manifest).python_functions == []
