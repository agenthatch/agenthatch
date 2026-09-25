"""Regression guard: locale-dependent file reads (v1.0.16).

Several files the CLI and core read are written as UTF-8 by this same
codebase (``init`` writes config.toml, ``hatch`` writes agenthatch.yaml,
the generator writes runtime.toml — all pass ``encoding="utf-8"``).  When
the *read* side omits the encoding, Python falls back to the platform
locale encoding, so the result silently depends on the machine:

- Chinese Windows: ``gbk`` / cp936 — any UTF-8 byte like ``→`` (E2 86 92)
  raises ``UnicodeDecodeError`` on the 0x92 continuation byte.
- ``LANG=C`` / ``LC_ALL=C`` (CI, Docker, cron, systemd): ``ascii``, which
  fails on *any* non-ASCII byte.

This bit ``agenthatch run`` for any generated ``agenthatch.yaml`` whose
skill text contained non-ASCII — a ``→`` in a weather-reporter "gotchas"
line was enough.

The writers are already UTF-8, so pinning the readers to UTF-8 restores
the read/write symmetry.  These guards keep it that way.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_CORE = REPO_ROOT / "agenthatch-core" / "src" / "agenthatch_core"
_SRC = REPO_ROOT / "src" / "agenthatch"

# Files whose content is UTF-8 on disk and whose readers must therefore be
# encoding-explicit.
UTF8_READ_FILES: list[Path] = [
    _CORE / "agent.py",                        # agenthatch.yaml (the `run` blocker)
    _CORE / "config.py",                       # ~/.agenthatch/config.toml
    _SRC / "cli" / "commands" / "assemble.py",  # agenthatch.yaml
    _SRC / "generate" / "engine.py",            # ~/.agenthatch/config.toml
    _SRC / "cli" / "commands" / "run.py",       # runtime.toml + config.toml
]


@pytest.mark.parametrize("path", UTF8_READ_FILES, ids=lambda p: p.name)
def test_no_bare_read_text(path: Path) -> None:
    """A bare ``read_text()`` makes the decoded text locale-dependent."""
    assert path.exists(), f"expected source file to exist: {path}"

    offenders = [
        f"{path.name}:{lineno}: {line.strip()}"
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        )
        if ".read_text()" in line
    ]

    assert not offenders, (
        "read_text() without encoding= is locale-dependent and breaks on "
        "non-UTF-8 locales (Chinese Windows / LANG=C):\n  "
        + "\n  ".join(offenders)
    )
