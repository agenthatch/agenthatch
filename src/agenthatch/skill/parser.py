"""Phase 1: Deterministic Context Assembly.

Three-step processing, zero AI participation:
  Step 1: Path resolution → dir_name
  Step 2: File discovery → FileManifest (SHA-256 + full text content)
  Step 3: YAML best-effort parsing → frontmatter dict | body raw

Phase 1 makes NO semantic classification of files.
That is LLM's responsibility (Phase 2 Harness).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import yaml

from agenthatch.skill.spec import ContextPack, FileEntry, FileManifest

# ── File reading constants ──────────────────────────────────────────────────
_MAX_FILE_CHARS = 10000
_MAX_FILE_BYTES = 1_000_000      # skip files > 1MB
_BINARY_HEAD_CHECK = 512

# Common binary file magic numbers
# PNG / JPEG / GIF / PDF / ZIP / RAR / GZIP / BZIP2 / Mach-O
_BIN_SIGS: list[bytes] = [
    b"\x89PNG\r\n\x1a\n",
    b"\xff\xd8\xff",
    b"GIF89a",
    b"GIF87a",
    b"%PDF",
    b"PK\x03\x04",
    b"Rar!\x1a\x07",
    b"\x1f\x8b",
    b"BZh",
    b"\xca\xfe\xba\xbe",
]

# SKILL.md case-insensitive matching
_SKILL_MD_VARIANTS = {"skill.md", "skil.md"}

# Directories to exclude during file discovery
_EXCLUDED_DIRS: frozenset[str] = frozenset({
    ".git", ".svn", "__pycache__", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "node_modules",
    ".venv", "venv", ".env", ".agenthatch",
})


def _is_skill_md(filename: str) -> bool:
    """Case-insensitive SKILL.md check.

    Matching filenames: SKILL.md, skill.md, Skill.md, SKILL.MD, skil.md, etc.
    """
    return filename.lower() in _SKILL_MD_VARIANTS


def assemble_context(skill_path: str | Path) -> ContextPack:
    """Phase 1 entry point: assemble deterministic context from a skill directory.

    Args:
        skill_path: Path to skill directory or SKILL.md file.

    Returns:
        ContextPack with frontmatter, body, file_manifest, dir_name, parse_warnings.

    Raises:
        FileNotFoundError: If the skill path does not exist.
    """
    # Step 1: Path resolution
    skill_dir = _resolve_skill_directory(Path(skill_path))
    dir_name = skill_dir.name

    # Step 2: File discovery + SHA-256 + full text content
    manifest = _discover_files(skill_dir)

    # Step 3: YAML best-effort parsing
    frontmatter, body, warnings = _best_effort_parse_yaml(skill_dir)

    return ContextPack(
        frontmatter=frontmatter,
        body=body,
        file_manifest=manifest,
        dir_name=dir_name,
        parse_warnings=warnings,
    )


def _resolve_skill_directory(path: Path) -> Path:
    """Resolve a path to a skill directory.

    If path is a .md file, return its parent directory.
    If path is a directory, return it directly.
    """
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Skill path not found: {path}")
    if path.is_file():
        if path.suffix.lower() in (".md", ".markdown"):
            return path.parent
        raise FileNotFoundError(f"Expected .md file or directory, got: {path}")
    return path


def _discover_files(skill_dir: Path) -> FileManifest:
    """Walk entire skill dir, collect ALL readable text files.

    BFS tree walk. We read file contents in addition to metadata,
    and we don't classify by extension.

    Returns:
        FileManifest with all entries populated. content=None for unreadable files.
    """
    manifest = FileManifest()

    for root, dirs, files in os.walk(skill_dir):
        # Prune excluded dirs in-place (os.walk convention)
        dirs[:] = [d for d in dirs if d not in _EXCLUDED_DIRS]

        for fname in files:
            fpath = Path(root) / fname
            rel = str(fpath.relative_to(skill_dir))
            st = fpath.stat()

            if st.st_size == 0:
                continue
            if st.st_size > _MAX_FILE_BYTES:
                continue

            entry = FileEntry(
                path=rel,
                hash=_sha256(fpath),
                size_bytes=st.st_size,
            )

            # Track entrypoint (case-insensitive)
            if _is_skill_md(fname):
                manifest.entrypoint = rel

            entry.content = _try_read_text(fpath)
            manifest.entries.append(entry)

    return manifest


def _try_read_text(filepath: Path) -> str | None:
    """Attempt to read file as text using binary-signature detection.

    Try decode, fail gracefully.
    Returns content string (truncated to _MAX_FILE_CHARS) or None.
    """
    # Stage 1: Read binary head for signature detection
    try:
        with open(filepath, "rb") as f:
            head = f.read(_BINARY_HEAD_CHECK)
    except OSError:
        return None

    if b"\x00" in head:
        return None

    for sig in _BIN_SIGS:
        if head.startswith(sig):
            return None

    # Stage 2: Attempt UTF-8 decode
    try:
        text = head.decode("utf-8")
    except UnicodeDecodeError:
        return None

    # Stage 3: Read remaining content
    remaining = filepath.stat().st_size - _BINARY_HEAD_CHECK
    if 0 < remaining < _MAX_FILE_BYTES:
        try:
            with open(filepath, "rb") as f:
                f.seek(_BINARY_HEAD_CHECK)
                rest = f.read(min(remaining, _MAX_FILE_CHARS)).decode(
                    "utf-8", errors="replace"
                )
                text += rest
        except (UnicodeDecodeError, OSError):
            pass

    # Stage 4: Truncate per-file
    if len(text) > _MAX_FILE_CHARS:
        text = text[:_MAX_FILE_CHARS] + "\n... (truncated)"

    return text


def _sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_markdown_file(skill_dir: Path) -> Path:
    """Find the primary Markdown file (SKILL.md or any .md)."""
    # Case-insensitive match first
    for entry in skill_dir.iterdir():
        if entry.is_file() and _is_skill_md(entry.name):
            return entry
    # Fallback: any .md file
    md_files = list(skill_dir.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No .md file found in {skill_dir}")
    return sorted(md_files)[0]


def _best_effort_parse_yaml(skill_dir: Path) -> tuple[dict[str, Any] | None, str, list[str]]:
    """Best-effort YAML frontmatter parsing.

    Strategy (descending priority):
      1. Standard ``---\\n...\\n---`` frontmatter → parse YAML
      2. ``---\\n...\\n---`` but bad YAML → record warning, frontmatter=None
      3. No ``---`` wrapper → frontmatter=None, body=full text
      4. ``---`` present but content not YAML → frontmatter=None, body=full text
      5. File not found → FileNotFoundError (blocks pipeline)

    Returns:
        (frontmatter, body_text, warning_list)
    """
    md_path = _find_markdown_file(skill_dir)
    raw = md_path.read_text(encoding="utf-8")

    # Detect YAML frontmatter boundaries: ^---$
    if not raw.startswith("---"):
        return None, raw, ["No YAML frontmatter detected"]

    # Find second ---
    parts = raw.split("\n", 1)
    if len(parts) < 2:
        return None, raw, ["Frontmatter delimiter found but no content"]

    rest = parts[1]
    # Find the closing ---
    idx = rest.find("\n---\n")
    if idx == -1:
        # Check if --- at end of file
        idx = rest.find("\n---")
        if idx == -1:
            # Special case: empty frontmatter (---\n---)
            if rest.startswith("---\n"):
                body_start = len("---\n")
                return {}, rest[body_start:].strip(), []
            if rest.startswith("---"):
                return {}, rest[len("---"):].strip(), []
            return None, raw, ["Unclosed frontmatter delimiter"]

    fm_text = rest[:idx]
    # Skip past closing delimiter to get body text
    if rest.startswith("\n---\n", idx):
        body_start = idx + len("\n---\n")
    else:
        body_start = idx + len("\n---")
    body_text = rest[body_start:].strip()

    try:
        import frontmatter
        post = frontmatter.loads(raw)
        if post.metadata and isinstance(post.metadata, dict):
            return dict(post.metadata), post.content, []
    except Exception:
        pass

    # Fallback: try yaml.safe_load directly
    try:
        metadata = yaml.safe_load(fm_text)
        if isinstance(metadata, dict):
            return metadata, body_text, []
        return None, body_text, ["Frontmatter parsed but is not a dict"]
    except yaml.YAMLError as e:
        return None, raw, [f"YAML parse error: {e}"]
    except Exception as e:
        return None, raw, [f"Unexpected parse error: {e}"]
