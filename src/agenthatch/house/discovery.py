"""Universal Skill Discovery (v0.7).

Scans multiple sources for skills, deduplicates by path, and
returns a unified catalog regardless of whether the skill lives
in ~/.claude/skills, ~/.codex/skills, ~/.agents/skills, or a
project-local .agents/skills directory.

Level 0 — no dependencies on agenthatch internals.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Known AI tool skill directories (ordered by priority)
KNOWN_SKILL_ROOTS: list[Path] = [
    # Primary — agenthatch native
    Path.home() / ".agents" / "skills",
    # Claude ecosystem
    Path.home() / ".claude" / "skills",
    # Codex ecosystem
    Path.home() / ".codex" / "skills",
    # OpenCode
    Path.home() / ".opencode" / "skills",
    # Continue.dev
    Path.home() / ".continue" / "skills",
    # Aider
    Path.home() / ".aider" / "skills",
    # Cline
    Path.home() / ".cline" / "skills",
    # Roo Code
    Path.home() / ".roo" / "skills",
    # Cursor
    Path.home() / ".cursor" / "skills",
    # Windsurf
    Path.home() / ".windsurf" / "skills",
    # Copilot / GitHub
    Path.home() / ".github-copilot" / "skills",
    Path.home() / ".copilot" / "skills",
    # Cody (Sourcegraph)
    Path.home() / ".cody" / "skills",
    # Tabby
    Path.home() / ".tabby" / "skills",
    # Goose
    Path.home() / ".goose" / "skills",
    # OpenHands
    Path.home() / ".openhands" / "skills",
    # GPT-Engineer
    Path.home() / ".gpt-engineer" / "skills",
    # Sweep
    Path.home() / ".sweep" / "skills",
    # Devin
    Path.home() / ".devin" / "skills",
    # Qodo
    Path.home() / ".qodo" / "skills",
    # Augment
    Path.home() / ".augment" / "skills",
    # Amazon Q
    Path.home() / ".amazonq" / "skills",
    # CodeRabbit
    Path.home() / ".coderabbit" / "skills",
    # Gemini CLI
    Path.home() / ".gemini" / "skills",
    # Qwen / Alibaba
    Path.home() / ".qwen" / "skills",
    # Replit
    Path.home() / ".replit" / "skills",
    # TaskMaster
    Path.home() / ".taskmaster" / "skills",
    # Open Interpreter
    Path.home() / ".open-interpreter" / "skills",
    # Kodu
    Path.home() / ".kodu" / "skills",
    # Pieces
    Path.home() / ".pieces" / "skills",
    # Ollama
    Path.home() / ".ollama" / "skills",
    # LM Studio
    Path.home() / ".lm-studio" / "skills",
    # Jan
    Path.home() / ".jan" / "skills",
]

# Directories to skip during scan
EXCLUDED_DIRS: frozenset[str] = frozenset({
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".tox", ".eggs", "dist", "build",
})

MAX_SCAN_DEPTH = 5
MAX_DIRS = 2000


@dataclass
class DiscoveredSkill:
    """A discovered skill from any source."""
    skill_id: str
    path: Path
    source: str           # "agents", "claude", "codex", "project", "custom"
    skill_md_path: Path | None = None
    ahs_path: Path | None = None
    has_scripts: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscoveryResult:
    """Result of universal skill discovery."""
    skills: list[DiscoveredSkill] = field(default_factory=list)
    sources_scanned: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.skills)

    def by_source(self, source: str) -> list[DiscoveredSkill]:
        return [s for s in self.skills if s.source == source]


def discover_all(
    extra_roots: list[Path] | None = None,
    is_skill_md: Callable[[str], bool] | None = None,
    max_depth: int = MAX_SCAN_DEPTH,
    max_dirs: int = MAX_DIRS,
) -> DiscoveryResult:
    """Discover all skills across known sources.

    Args:
        extra_roots: Additional search roots beyond the builtin set.
        is_skill_md: Predicate to identify SKILL.md files (default: exact match).
        max_depth: Maximum directory depth for BFS scan.
        max_dirs: Maximum directories to visit per root.

    Returns:
        DiscoveryResult with deduplicated skills list.
    """
    if is_skill_md is None:
        is_skill_md = _default_skill_md_matcher

    result = DiscoveryResult()
    seen: set[Path] = set()
    roots: list[tuple[Path, str]] = []

    # Known roots
    for root in KNOWN_SKILL_ROOTS:
        roots.append((root, _root_source_name(root)))

    # Extra roots from config or CLI
    for root in (extra_roots or []):
        roots.append((root, "custom"))

    # Project-local root
    project_root = Path.cwd() / ".agents" / "skills"
    if project_root.is_dir():
        roots.append((project_root, "project"))

    for root, source in roots:
        if not root.is_dir():
            continue
        result.sources_scanned.append(f"{source}:{root}")

        try:
            root_skills = _scan_root(
                root, source, is_skill_md, seen, max_depth, max_dirs,
            )
            result.skills.extend(root_skills)
        except Exception as e:
            msg = f"Error scanning {root}: {e}"
            logger.warning(msg)
            result.errors.append(msg)

    return result


def _root_source_name(root: Path) -> str:
    """Infer source name from known paths."""
    root_str = str(root)
    if ".agents" in root_str:
        return "agents"
    if ".claude" in root_str:
        return "claude"
    if ".codex" in root_str:
        return "codex"
    return "custom"


def _default_skill_md_matcher(filename: str) -> bool:
    """Default SKILL.md matcher — case-insensitive exact match."""
    return filename.upper() == "SKILL.MD"


def _scan_root(
    root: Path,
    source: str,
    is_skill_md: Callable[[str], bool],
    seen: set[Path],
    max_depth: int,
    max_dirs: int,
) -> list[DiscoveredSkill]:
    """BFS scan a single root for skills."""
    skills: list[DiscoveredSkill] = []
    visited: set[Path] = set()
    queue: deque[tuple[Path, int]] = deque()

    try:
        root = root.resolve(strict=True)
    except (OSError, FileNotFoundError):
        return skills

    visited.add(root)
    queue.append((root, 0))
    dirs_visited = 0

    while queue:
        current_dir, depth = queue.popleft()
        dirs_visited += 1
        if dirs_visited > max_dirs:
            logger.warning("Scan truncated at %d dirs in %s", max_dirs, root)
            break

        try:
            entries = list(current_dir.iterdir())
        except (OSError, PermissionError):
            continue

        has_skill_md = False
        skill_md_path: Path | None = None
        ahs_path: Path | None = None
        has_scripts = False
        subdirs: list[Path] = []

        for entry in entries:
            if entry.is_symlink():
                continue

            if entry.is_file():
                name = entry.name
                if is_skill_md(name):
                    has_skill_md = True
                    skill_md_path = entry
                elif name == "agenthatch.yaml":
                    ahs_path = entry
            elif entry.is_dir() and depth < max_depth:
                if not entry.name.startswith(".") and entry.name not in EXCLUDED_DIRS:
                    resolved = entry.resolve()
                    if resolved not in visited:
                        subdirs.append(resolved)
                        if entry.name == "scripts":
                            has_scripts = True

        if has_skill_md:
            resolved = current_dir.resolve()
            if resolved not in seen:
                seen.add(resolved)
                skills.append(DiscoveredSkill(
                    skill_id=current_dir.name,
                    path=resolved,
                    source=source,
                    skill_md_path=skill_md_path,
                    ahs_path=ahs_path,
                    has_scripts=has_scripts,
                ))

        for subdir in subdirs:
            visited.add(subdir)
            queue.append((subdir, depth + 1))

    return skills
