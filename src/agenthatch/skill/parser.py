"""Phase 1: Deterministic Context Assembly + Phase 1.5 ScriptAnalyzer.

Three-step processing, zero AI participation:
  Step 1: Path resolution → dir_name
  Step 2: File discovery → FileManifest (SHA-256 + full text content)
  Step 3: YAML best-effort parsing → frontmatter dict | body raw

Phase 1 makes NO semantic classification of files.
That is LLM's responsibility (Phase 2 Harness).

v0.8: Phase 1.5 ScriptAnalyzer adds deterministic AST parsing of Python
scripts and regex parsing of shell scripts to extract function signatures.
This feeds into Harness C for precise interface inference.
"""

from __future__ import annotations

import ast as _ast
import hashlib
import os
import re as _re
from dataclasses import dataclass
from dataclasses import field as _dc_field
from pathlib import Path
from typing import Any

import yaml

from agenthatch.skill.spec import ContextPack, FileEntry, FileManifest

# ── File reading constants ──────────────────────────────────────────────────
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
        skill_dir=skill_dir,  # v0.8: for Phase 1.5 ScriptAnalyzer
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
                rest = f.read(remaining).decode(
                    "utf-8", errors="replace"
                )
                text += rest
        except (UnicodeDecodeError, OSError):
            pass

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


# ─────────────────────────────────────────────────────────────────────────
# v0.8: Phase 1.5 ScriptAnalyzer — deterministic AST/regex analysis
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class ToolSchema:
    """Deterministically extracted tool signature from a script."""
    name: str
    args: list[dict[str, str | None]] = _dc_field(default_factory=list)
    returns: str | None = None
    docstring: str | None = None
    source_file: str = ""


@dataclass
class ScriptManifest:
    """Output of Phase 1.5 ScriptAnalyzer.

    Contains deterministically extracted function signatures from all
    scripts in the skill directory. Fed into Harness C for precise
    interface inference instead of raw file content.
    """
    python_functions: list[ToolSchema] = _dc_field(default_factory=list)
    shell_functions: list[dict[str, str]] = _dc_field(default_factory=list)
    has_binary_assets: list[str] = _dc_field(default_factory=list)
    asset_metadata: list[dict[str, Any]] = _dc_field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.python_functions and not self.shell_functions


def extract_python_signatures(file_path: Path) -> list[ToolSchema]:
    """AST-parse a Python script, extract public function signatures.

    Deterministic, zero LLM. Uses Python's built-in ``ast`` module.
    Skips private functions (those starting with ``_``).

    Returns:
        List of ToolSchema, one per public function found.
    """
    try:
        tree = _ast.parse(file_path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError, OSError) as e:
        import logging
        logging.getLogger("agenthatch").warning(
            "ScriptAnalyzer: cannot parse %s: %s", file_path, e
        )
        return []

    functions: list[ToolSchema] = []
    for node in tree.body:
        if not isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_"):
            continue
        # Process this top-level function (not nested in class/function)
            args: list[dict[str, str | None]] = []
            for arg in node.args.args:
                arg_type: str | None = None
                if arg.annotation:
                    try:
                        arg_type = _ast.unparse(arg.annotation)
                    except Exception:
                        arg_type = None
                args.append({"name": arg.arg, "type": arg_type})

            returns: str | None = None
            if node.returns:
                try:
                    returns = _ast.unparse(node.returns)
                except Exception:
                    returns = None

            functions.append(ToolSchema(
                name=node.name,
                args=args,
                returns=returns,
                docstring=_ast.get_docstring(node),
                source_file=str(file_path),
            ))
    return functions


def extract_shell_functions(file_path: Path) -> list[dict[str, str]]:
    """Regex-parse shell scripts for function definitions.

    Matches both ``function name()`` and ``name()`` syntax.
    """
    try:
        content = file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []

    functions: list[dict[str, str]] = []
    pattern = _re.compile(r'^(?:function\s+)?(\w+)\s*\(\s*\)', _re.MULTILINE)
    for match in pattern.finditer(content):
        functions.append({
            "name": match.group(1),
            "source_file": str(file_path),
        })
    return functions


def analyze_scripts(skill_dir: Path) -> ScriptManifest:
    """Phase 1.5 entry point: analyze all scripts/ in a skill directory.

    Walks the ``skills/scripts/`` subdirectory and extracts function
    signatures from all Python (.py) and shell (.sh) files.

    Args:
        skill_dir: Path to the skill directory (contains skills/scripts/).

    Returns:
        ScriptManifest with all extracted function signatures.
    """
    manifest = ScriptManifest()
    scripts_dir = skill_dir / "skills" / "scripts"
    if not scripts_dir.is_dir():
        return manifest

    for script_file in sorted(scripts_dir.iterdir()):
        if not script_file.is_file():
            continue
        if script_file.suffix == ".py":
            manifest.python_functions.extend(
                extract_python_signatures(script_file)
            )
        elif script_file.suffix == ".sh":
            manifest.shell_functions.extend(
                extract_shell_functions(script_file)
            )
    return manifest


def analyze_scripts_from_manifest(file_manifest: FileManifest) -> ScriptManifest:
    """Analyze scripts from a FileManifest (in-memory, no disk access).

    Used when script content is already loaded in FileManifest entries.
    Scans entries with .py/.sh suffixes and attempts AST/regex extraction
    from their content strings.
    """
    manifest = ScriptManifest()
    for entry in file_manifest.entries:
        if entry.content is None:
            continue
        suffix = Path(entry.path).suffix.lower()
        if suffix == ".py":
            try:
                tree = _ast.parse(entry.content)
            except SyntaxError:
                continue
            for node in tree.body:
                if not isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                    continue
                if node.name.startswith("_"):
                    continue
                    args: list[dict[str, str | None]] = []
                    for arg in node.args.args:
                        arg_type: str | None = None
                        if arg.annotation:
                            try:
                                arg_type = _ast.unparse(arg.annotation)
                            except Exception:
                                arg_type = None
                        args.append({"name": arg.arg, "type": arg_type})
                    returns: str | None = None
                    if node.returns:
                        try:
                            returns = _ast.unparse(node.returns)
                        except Exception:
                            returns = None
                    manifest.python_functions.append(ToolSchema(
                        name=node.name,
                        args=args,
                        returns=returns,
                        docstring=_ast.get_docstring(node),
                        source_file=entry.path,
                    ))
        elif suffix == ".sh":
            pattern = _re.compile(r'^(?:function\s+)?(\w+)\s*\(\s*\)', _re.MULTILINE)
            for match in pattern.finditer(entry.content):
                manifest.shell_functions.append({
                    "name": match.group(1),
                    "source_file": entry.path,
                })
    return manifest


def format_script_manifest(manifest: ScriptManifest) -> str:
    """Format ScriptManifest for LLM consumption in Harness C.

    Produces a compact summary (< 1KB typical) with all function
    signatures, types, and docstrings. Harness C uses this instead
    of raw file content for precise interface inference.
    """
    if manifest.is_empty():
        return "(no script signatures extracted)"

    lines: list[str] = ["## Extracted Script Signatures (deterministic)\n"]

    if manifest.python_functions:
        lines.append("### Python Functions\n")
        for func in manifest.python_functions:
            args_str = ", ".join(
                f"{a['name']}: {a['type'] or 'Any'}" for a in func.args
            )
            returns = f" → {func.returns}" if func.returns else ""
            lines.append(f"- `{func.name}({args_str})`{returns}")
            if func.docstring:
                doc = func.docstring[:120].replace("\n", " ")
                lines.append(f"  > {doc}")

    if manifest.shell_functions:
        lines.append("\n### Shell Functions\n")
        for func in manifest.shell_functions:  # type: ignore[assignment]
            lines.append(f"- `{func['name']}()` (shell)")  # type: ignore[index]

    return "\n".join(lines)


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
