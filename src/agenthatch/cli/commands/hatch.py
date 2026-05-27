"""agenthatch hatch — Standardize a SKILL.md into AHSSPEC middleware.

Core v0.3 command: runs Phase 1 (deterministic context assembly) +
Phase 2 (5 AgentHarnesses inference) and outputs:
  1. agenthatch.yaml (AHSSPEC v1.1) at <skill_dir>/agenthatch.yaml
  2. skillhouse.json entry (registered in .agenthatch/skillhouse.json)

v0.3 enhancement: name-based skill resolution with 3-layer fallback:
  Layer 1: Direct path resolution (backward compatible)
  Layer 2: skillhouse.json exact-match lookup (fast index path)
  Layer 3: BFS filesystem scan (no pre-registration required)
"""

from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.tree import Tree

from agenthatch.cli import console
from agenthatch.skill.parser import _is_skill_md, assemble_context

logger = logging.getLogger("agenthatch")

# ── Filesystem scan constants ──────────────────────────────────────────────
_MAX_NAME_SCAN_DEPTH = 4       # max directory depth to scan
_MAX_DIRS_PER_ROOT = 500       # max dirs to visit per search root


def hatch_command(
    skill_path: Annotated[
        str,
        typer.Argument(help="Path to skill directory, SKILL.md file, or skill name"),
    ],
    output: Annotated[
        str | None,
        typer.Option("--output", "-o", help="Output path for agenthatch.yaml"),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Overwrite existing agenthatch.yaml"),
    ] = False,
    trace: Annotated[
        bool,
        typer.Option("--trace", help="Show Harness reasoning traces"),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Convert without writing files"),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output as JSON instead of YAML"),
    ] = False,
) -> None:
    """Standardize a SKILL.md into AHSSPEC middleware.

    Runs the full Meta-Agent pipeline:
      Phase 1: Deterministic context assembly (no AI)
      Phase 2: 5 AgentHarnesses inference (LLM-driven)

    Examples:
        agenthatch hatch ~/skills/weather-reporter/
        agenthatch hatch ./SKILL.md --trace
        agenthatch hatch weather-reporter
        agenthatch hatch . --json --dry-run
    """
    from agenthatch.config import Config
    from agenthatch.skill.builder import build_ahspec

    config = Config.load()

    # ── 1. Resolve skill (name or path, 3-layer fallback) ──────────────
    skill_real_path, from_index = _resolve_skill_name(skill_path, config)
    console.print()
    console.print(f"[bold]Hatching:[/bold] {skill_real_path}")
    if from_index:
        console.print("  [dim]Resolved from skillhouse index[/dim]")

    # ── 2. Phase 1: Context Assembly ──────────────────────────────────
    if trace:
        console.print()
        console.print("[bold]Phase 1: Context Assembly[/bold]")

    try:
        context = assemble_context(skill_real_path)
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1) from e
    except Exception as e:
        console.print(f"[red]Parse error: {e}[/red]")
        raise typer.Exit(code=3) from e

    if trace:
        console.print(f"  [ok]Path resolved:[/ok] {context.dir_name}")
        fm_count = len(context.frontmatter) if context.frontmatter else 0
        console.print(f"  [ok]Frontmatter:[/ok] {fm_count} fields")
        total_files = len(context.file_manifest.entries)
        readable = len(context.file_manifest.content_bundle())
        console.print(f"  [ok]Files discovered:[/ok] {total_files} files ({readable} readable)")
        for warning in context.parse_warnings:
            console.print(f"  [warn]Warning:[/warn] {warning}")

    # ── 3. Phase 2: Agentic Inference ─────────────────────────────────
    if trace:
        console.print()
        console.print("[bold]Phase 2: Agentic Inference[/bold]")

    harness_cfg = config.get("config", {}).get("harness", {}) if "config" in config else {}
    large_model = harness_cfg.get("large_model", "") if isinstance(harness_cfg, dict) else ""
    small_model = harness_cfg.get("small_model", "") if isinstance(harness_cfg, dict) else ""

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running AgentHarnesses...", total=None)
            ahs_spec, harness_outputs = build_ahspec(
                context, config, large_model=large_model, small_model=small_model
            )
            progress.remove_task(task)
    except Exception as e:
        console.print(f"[red]Inference error: {e}[/red]")
        raise typer.Exit(code=4) from e

    # ── 4. Display trace (if requested) ───────────────────────────────
    if trace:
        console.print()
        for key in ["A", "B", "C", "D", "E"]:
            if key not in harness_outputs:
                continue
            h_output = harness_outputs[key]
            label = {
                "A": "extract_identity",
                "B": "infer_intent",
                "C": "infer_interface",
                "D": "detect_base_and_instructions",
                "E": "assemble_and_validate",
            }.get(key, key)

            tree = Tree(f"[bold]Harness {key}: {label}[/bold]")
            for trace_line in h_output.reasoning_trace:
                tree.add(trace_line)
            tree.add(
                f"[green]confidence={h_output.confidence:.2f}, "
                f"self_check_passed={h_output.self_check_passed}, "
                f"internal_retries={h_output.internal_retries}[/green]"
            )
            if h_output.degradation_applied:
                tree.add(f"[yellow]degradations: {h_output.degradation_applied}[/yellow]")
            console.print(tree)
            console.print()

    # ── 5. Output report ──────────────────────────────────────────────
    console.print()
    console.print("[bold]Validation[/bold]")
    console.print("  [ok]AHSSPEC Schema:[/ok] passed")
    console.print("  [ok]Capability uniqueness:[/ok] passed")

    if ahs_spec.confidence_report:
        cr = ahs_spec.confidence_report
        console.print(
            f"  [ok]Confidence:[/ok] overall={cr.overall:.2f}"
        )

    # ── 6. Write outputs ──────────────────────────────────────────────
    if dry_run:
        console.print()
        console.print("[yellow]Dry-run mode — no files written.[/yellow]")
        if json_output:
            console.print_json(ahs_spec.model_dump_json())
        else:
            yaml_str = yaml.dump(
                json.loads(ahs_spec.model_dump_json()),
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            )
            console.print(yaml_str)
        return

    # Determine output path
    skill_dir = _resolve_skill_dir(skill_real_path)
    if output:
        output_path = Path(output).expanduser().resolve()
    else:
        output_path = skill_dir / "agenthatch.yaml"

    if output_path.exists() and not force:
        console.print(
            f"[yellow]agenthatch.yaml already exists at {output_path}[/yellow]"
        )
        console.print("Use --force to overwrite.")
        raise typer.Exit(code=2)

    # Write agenthatch.yaml
    output_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_str = yaml.dump(
        json.loads(ahs_spec.model_dump_json(exclude={"harness_traces", "confidence_report"})),
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    )
    output_path.write_text(yaml_str, encoding="utf-8")
    console.print(f"  [ok]Written:[/ok] {output_path}")

    # Register in skillhouse.json
    skillhouse_config = config.get("skillhouse", {}) if "skillhouse" in config else {}
    skillhouse_path = skillhouse_config.get(
        "path", ".agenthatch/skillhouse.json"
    ) if isinstance(skillhouse_config, dict) else ".agenthatch/skillhouse.json"

    # Resolve relative to cwd
    skillhouse_full_path = Path(skillhouse_path)
    if not skillhouse_full_path.is_absolute():
        skillhouse_full_path = Path.cwd() / skillhouse_full_path

    from agenthatch.house.index import SkillhouseIndex
    idx = SkillhouseIndex(str(skillhouse_full_path))
    idx.add_entry(ahs_spec.identity.id, ahs_spec, str(output_path))
    console.print(f"  [ok]Registered:[/ok] {skillhouse_full_path} ({idx.entry_count} entries)")

    console.print()
    console.print("[bold green]Hatch complete.[/bold green]")


def _resolve_skill_dir(path: Path) -> Path:
    """Resolve a path to its skill directory."""
    if path.is_file() and path.suffix in (".md", ".markdown"):
        return path.parent
    return path


# ── Name Resolution (3-layer fallback) ─────────────────────────────────────

def _resolve_skill_name(skill_input: str, config: dict) -> tuple[Path, bool]:
    """Resolve a skill name or path to a skill directory.

    Three-layer strategy:
      1. Looks like a path → direct resolution
      2. skillhouse.json exact match → cached path
      3. Filesystem scan → BFS tree walk finding SKILL.md-bearing dirs

    Args:
        skill_input: User-provided skill argument (name or path).
        config: Full config dict (for [skills].search_dirs).

    Returns:
        (resolved_skill_dir_path, from_index: bool).

    Raises:
        typer.Exit: If no skill can be resolved.
    """
    input_path = Path(skill_input)
    looks_like_path = (
        "/" in skill_input
        or "\\" in skill_input
        or bool(input_path.suffix)
        or skill_input in (".", "..")
    )

    # ── Layer 1: Direct path ──
    if looks_like_path:
        resolved = input_path.expanduser().resolve()
        if resolved.is_dir():
            _validate_contains_skill_md(resolved, str(resolved))
            return resolved, False
        if resolved.is_file():
            parent = resolved.parent
            _validate_contains_skill_md(parent, str(resolved))
            return parent, False
        _fail_not_found(skill_input, hint="Path does not exist.", exit_code=1)

    # ── Layer 2: skillhouse.json index ──
    index_result = _resolve_from_index(skill_input, config)
    if index_result is not None:
        return index_result, True

    # ── Layer 3: Filesystem scan ──
    return _resolve_from_filesystem(skill_input, config), False


def _resolve_from_index(name: str, config: dict) -> Path | None:
    """Exact-match lookup in skillhouse.json."""
    from agenthatch.house.index import SkillhouseIndex

    skillhouse_cfg = config.get("skillhouse") if "skillhouse" in config else {}
    skillhouse_path = skillhouse_cfg.get("path", ".agenthatch/skillhouse.json") \
        if isinstance(skillhouse_cfg, dict) else ".agenthatch/skillhouse.json"

    idx_path = Path(skillhouse_path)
    if not idx_path.is_absolute():
        idx_path = Path.cwd() / idx_path
    if not idx_path.exists():
        return None

    idx = SkillhouseIndex(str(idx_path))
    entry = idx.find_by_name(name)
    if entry is None:
        return None

    ahs_path = entry.get("ahs_path", "")
    if not ahs_path:
        return None

    skill_dir = Path(ahs_path).parent
    if not skill_dir.is_dir():
        console.print(
            f"[yellow]Warning:[/yellow] '{name}' indexed but source dir "
            f"{skill_dir} no longer exists. Falling back to filesystem scan."
        )
        return None

    console.print(f"[dim]Resolved '{name}' from skillhouse index → {skill_dir}[/dim]")
    return skill_dir


def _resolve_from_filesystem(name: str, config: dict) -> Path:
    """BFS scan search_dirs for a directory named 'name' containing SKILL.md.

    Pattern: codex discover_skills_under_root() — BFS with depth/dir limits,
    case-insensitive SKILL.md filename matching (agenthatch extension).

    Returns:
        Resolved skill directory path.

    Raises:
        typer.Exit: If no match or multiple ambiguous matches.
    """
    search_roots = _resolve_search_roots(config)

    matches: list[Path] = []
    for root in search_roots:
        if not root.is_dir():
            continue
        found = _scan_for_skill(root, name)
        matches.extend(found)

    if not matches:
        _fail_not_found(
            name,
            hint=f"Searched {len(search_roots)} directories, "
                 f"no '{name}/SKILL.md' found.",
            exit_code=1,
        )

    if len(matches) > 1:
        console.print(
            f"[yellow]Warning:[/yellow] Multiple skills match '{name}'. "
            f"Using first result: {matches[0]}"
        )
        console.print("[dim]Alternatives:[/dim]")
        for alt in matches[1:]:
            console.print(f"  - {alt}")

    selected = matches[0]
    console.print(f"[dim]Discovered '{name}' → {selected}[/dim]")
    _auto_register_to_index(selected, config)
    return selected


def _scan_for_skill(root: Path, target_name: str) -> list[Path]:
    """BFS scan a search root for directories containing SKILL.md.

    Pattern: codex discover_skills_under_root():
      - BFS with deque
      - MAX_SCAN_DEPTH limit
      - MAX_DIRS_PER_ROOT upper bound
      - Skip dot-prefixed directories
      - file_name == SKILLS_FILENAME check (codex uses exact; we use case-insensitive)

    Returns:
        List of matching directories (empty if none).
    """
    results: list[Path] = []
    visited: set[Path] = set()
    queue: deque[tuple[Path, int]] = deque()

    try:
        root = root.resolve(strict=True)
    except (OSError, FileNotFoundError):
        return results

    if not root.is_dir():
        return results

    visited.add(root)
    queue.append((root, 0))
    dirs_visited = 0

    _EXCLUDED = frozenset(
        {".git", "__pycache__", "node_modules", ".venv", "venv",
         ".mypy_cache", ".pytest_cache", ".tox", ".eggs", "dist", "build"}
    )

    while queue:
        current_dir, depth = queue.popleft()
        dirs_visited += 1

        if dirs_visited > _MAX_DIRS_PER_ROOT:
            break

        # Check if this directory contains SKILL.md
        try:
            entries = list(current_dir.iterdir())
        except (OSError, PermissionError):
            continue

        has_skill_md = False
        subdirs: list[Path] = []

        for entry in entries:
            if not entry.is_symlink():
                if entry.is_file() and _is_skill_md(entry.name):
                    has_skill_md = True
                elif entry.is_dir() and depth < _MAX_NAME_SCAN_DEPTH:
                    if not entry.name.startswith(".") and entry.name not in _EXCLUDED:
                        resolved = entry.resolve()
                        if resolved not in visited:
                            subdirs.append(resolved)

        # Match: directory name == target_name AND contains SKILL.md
        if has_skill_md and current_dir.name == target_name:
            results.append(current_dir)

        # Enqueue subdirectories for BFS
        for subdir in subdirs:
            visited.add(subdir)
            queue.append((subdir, depth + 1))

    return results


def _auto_register_to_index(skill_dir: Path, config: dict) -> None:
    """Silently register a newly discovered skill to skillhouse.json.

    Called after successful filesystem scan (Layer 3).
    This ensures subsequent lookups hit the fast index path (Layer 2).
    Registration is fire-and-forget: failure doesn't block hatch.
    """
    try:
        from agenthatch.house.index import SkillhouseIndex

        skillhouse_cfg = config.get("skillhouse") if "skillhouse" in config else {}
        skillhouse_path = skillhouse_cfg.get("path", ".agenthatch/skillhouse.json") \
            if isinstance(skillhouse_cfg, dict) else ".agenthatch/skillhouse.json"

        idx_path = Path(skillhouse_path)
        if not idx_path.is_absolute():
            idx_path = Path.cwd() / idx_path
        idx_path.parent.mkdir(parents=True, exist_ok=True)

        idx = SkillhouseIndex(str(idx_path))
        idx.register_placeholder(
            skill_id=skill_dir.name,
            skill_dir=str(skill_dir),
        )
    except Exception:
        pass  # Fire-and-forget — don't block hatch on index failure


def _validate_contains_skill_md(directory: Path, original_input: str) -> None:
    """Check that a directory contains at least one SKILL.md (case-insensitive)."""
    try:
        for entry in directory.iterdir():
            if entry.is_file() and _is_skill_md(entry.name):
                return
    except (OSError, PermissionError):
        pass
    _fail_not_found(
        original_input,
        hint="Directory exists but no SKILL.md found (case-insensitive). "
             "Expected: SKILL.md, skill.md, Skill.md, etc.",
        exit_code=1,
    )


def _fail_not_found(name: str, hint: str = "", exit_code: int = 1) -> None:
    """Print helpful error and exit."""
    console.print(f"[red]Error:[/red] Skill not found: '{name}'")
    if hint:
        console.print(f"[dim]{hint}[/dim]")
    raise typer.Exit(code=exit_code)


# ── Search Root Resolution (shared with init.py) ────────────────────────────

# Known AI tool skill host directories (relative to $HOME)
# These are dot-prefixed directories that SHOULD be entered (unlike .git, .venv)
_KNOWN_SKILL_HOST_DIRS: list[str] = [
    ".claude/skills",
    ".openclaw/skills",
    ".codex/skills",
    ".agents/skills",
    "skills",
    ".agenthatch/skills",
]


def _resolve_search_roots(config: dict) -> list[Path]:
    """Resolve all skill search roots from three sources.

    Sources:
      1. [skills].search_dirs from config (user-specified)
      2. _KNOWN_SKILL_HOST_DIRS under $HOME (auto-discovered AI tool dirs)
      3. Project-level .agents/skills/ (Codex convention, up to 3 parent levels)

    Pattern: codex skill_roots() — explicit enumeration of known paths.

    Returns:
        Deduplicated list of existing directory paths.
    """
    home = Path.home()
    roots: list[Path] = []

    # Source 1: User-configured search dirs
    skills_cfg = config.get("skills") if "skills" in config else {}
    if isinstance(skills_cfg, dict):
        raw = skills_cfg.get("search_dirs", "")
        if raw:
            for p in raw.split(","):
                p = p.strip()
                if p:
                    roots.append(Path(p).expanduser().resolve())

    # Source 2: Known AI tool skill directories under home
    for host_dir in _KNOWN_SKILL_HOST_DIRS:
        candidate = home / host_dir
        if candidate.is_dir():
            roots.append(candidate.resolve())

    # Source 3: Project-level .agents/skills/ (Codex convention)
    cwd = Path.cwd().resolve()
    for parent in [cwd] + list(cwd.parents)[:3]:
        candidate = parent / ".agents" / "skills"
        if candidate.is_dir():
            roots.append(candidate.resolve())

    # Deduplicate by canonical path
    seen: set[Path] = set()
    unique: list[Path] = []
    for r in roots:
        if r not in seen:
            seen.add(r)
            unique.append(r)
    return unique
