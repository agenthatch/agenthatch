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
import time
from collections import deque
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree

from agenthatch.cli import console
from agenthatch.skill.parser import _is_skill_md, assemble_context
from agenthatch.skill.spec import AgentConfig

logger = logging.getLogger("agenthatch")

# ── Filesystem scan constants ──────────────────────────────────────────────
_MAX_NAME_SCAN_DEPTH = 4       # max directory depth to scan
_MAX_DIRS_PER_ROOT = 500       # max dirs to visit per search root

# ── Phase 3 agent generation ───────────────────────────────────────────────


def _run_phase3_generate(
    ahs_spec: Any,
    skill_dir: Path,
    output: str | None,
    force: bool,
    dry_run: bool,
    copy_skills: bool,
    _framework: str,
) -> tuple[int, Path]:
    """Run Phase 3: generate an independent Agent directory from AHSSPEC.

    Returns:
        (file_count, agent_output_dir)
    """
    import json as _json

    from agenthatch.generate.engine import generate_agent

    agent_id = ahs_spec.identity.id
    if output:
        agent_output_dir = Path(output).expanduser().resolve()
    else:
        agent_output_dir = Path.cwd() / f"{agent_id}-agent"

    spec_dict = _json.loads(ahs_spec.model_dump_json())

    try:
        written = generate_agent(
            ahspec=spec_dict,
            output_dir=agent_output_dir,
            dry_run=dry_run,
            force=force,
            copy_skills=copy_skills,
            skill_dir=skill_dir,
        )
    except FileExistsError as e:
        console.print(f"[yellow]{e}[/yellow]")
        console.print("Use --force to overwrite.")
        raise typer.Exit(code=2) from e
    except Exception as e:
        console.print(f"[error]Generation error: {e}[/error]")
        raise typer.Exit(code=5) from e

    return len(written), agent_output_dir


# ── CLI rendering helpers ───────────────────────────────────────────────────


def _format_size(size_bytes: int) -> str:
    """Format byte count as human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def _render_confidence(ahs_spec: Any) -> None:
    """Render confidence panel with bar chart for each harness."""
    cr = ahs_spec.confidence_report
    if not cr or not cr.per_harness:
        return

    labels = {
        "A": "extract_identity",
        "B": "infer_intent",
        "C": "infer_interface",
        "D": "detect_base",
        "E": "assemble_validate",
    }

    BAR_WIDTH = 22
    mcp_servers = (
        ahs_spec.interface.mcp_servers
        if getattr(ahs_spec.interface, "mcp_servers", None)
        else []
    )

    lines: list[Text] = []
    for key in ["A", "B", "C", "D", "E"]:
        if key not in cr.per_harness:
            continue
        name = labels.get(key, key)
        score = cr.per_harness[key]
        filled = max(1, int(score * BAR_WIDTH))

        line = Text()
        line.append("  ")
        line.append(key, style="accent")
        line.append("  ")
        line.append(f"{name:<24}", style="dim")
        line.append("  ")

        bar = Text("▓" * filled + "░" * (BAR_WIDTH - filled))
        bar.stylize("ok", 0, filled)
        bar.stylize("dim", filled, BAR_WIDTH)
        line.append_text(bar)

        line.append(f"  {score:.2f}")

        if key == "D" and mcp_servers:
            s = "s" if len(mcp_servers) > 1 else ""
            line.append(f"  mcp: {len(mcp_servers)} server{s}", style="dim")

        lines.append(line)

    console.print()
    console.print(Panel(Text("\n").join(lines), title="Confidence", border_style="cyan"))


def _render_harness_traces(harness_outputs: dict[str, Any]) -> None:
    """Render Rich Tree for each harness trace (--trace mode)."""
    labels = {
        "A": "extract_identity",
        "B": "infer_intent",
        "C": "infer_interface",
        "D": "detect_base_and_instructions",
        "E": "assemble_and_validate",
    }
    for key in ["A", "B", "C", "D", "E"]:
        if key not in harness_outputs:
            continue
        h_output = harness_outputs[key]
        label = labels.get(key, key)

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


def _render_phase3_result(
    file_count: int,
    agent_output_dir: Path,
    elapsed: float,
    dry_run: bool,
    trace: bool,
    written_files: list[Path] | None = None,
) -> None:
    """Render Phase 3 completion output."""
    if dry_run:
        console.print(
            f"[dim]◌  Would generate {file_count} files → {agent_output_dir}[/dim]"
        )
    else:
        console.print(
            f"[ok]✓[/ok]  {file_count} files generated in {elapsed:.1f} seconds"
        )
        if trace and written_files:
            for f in sorted(written_files):
                try:
                    rel = f.relative_to(agent_output_dir)
                except ValueError:
                    rel = f
                console.print(f"    [dim]{rel}[/dim]")


# ── Main hatch command ──────────────────────────────────────────────────────


def hatch_command(
    skill_path: Annotated[
        str,
        typer.Argument(help="Path to skill directory, SKILL.md file, or skill name"),
    ],
    output: Annotated[
        str | None,
        typer.Option(
            "--output", "-o",
            help="Agent output directory (default: ./<skill-id>-agent/)",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Overwrite existing output directory"),
    ] = False,
    trace: Annotated[
        bool,
        typer.Option("--trace", help="Show Harness reasoning traces"),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print generated files without writing"),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output as JSON instead of YAML"),
    ] = False,
    no_generate: Annotated[
        bool,
        typer.Option(
            "--no-generate",
            help="Skip Phase 3 — only produce agenthatch.yaml (review mode)",
        ),
    ] = False,
    no_copy_skills: Annotated[
        bool,
        typer.Option("--no-copy-skills", help="Exclude original SKILL.md and resource files"),
    ] = False,
    framework: Annotated[
        str,
        typer.Option("--framework", help="Agent framework [python-typer]"),
    ] = "python-typer",
) -> None:
    """Standardize a SKILL.md into AHSSPEC middleware and generate an independent Agent.

    v0.6: hatch now runs the full three-phase pipeline by default:
      Phase 1: Deterministic context assembly (no AI)
      Phase 2: 5 AgentHarnesses inference (LLM-driven)
      Phase 3: Agent generation via Jinja2 templates (default on)

    Examples:
        agenthatch hatch ~/skills/weather-reporter/
        agenthatch hatch ./SKILL.md --trace
        agenthatch hatch weather-reporter
        agenthatch hatch . --no-generate        # review mode: yaml only
        agenthatch hatch . --dry-run            # preview without writing
    """
    from agenthatch.config import Config
    from agenthatch.skill.builder import build_ahspec

    # Suppress third-party log noise leaking to CLI output
    for noisy in (
        "sentence_transformers",
        "huggingface_hub",
        "urllib3",
        "urllib3.connectionpool",
        "urllib3.util.retry",
        "httpx",
        "httpcore",
        "openai",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    config = Config.load()

    # ── 1. Resolve skill (name or path, 3-layer fallback) ──────────────
    skill_real_path, from_index = _resolve_skill_name(skill_path, config)

    # ── 2. Early output dir resolution (for overview + conflict check) ─
    skill_dir = _resolve_skill_dir(skill_real_path)
    skill_name = skill_dir.name
    if output:
        agent_output_dir_early = Path(output).expanduser().resolve()
    else:
        agent_output_dir_early = Path.cwd() / f"{skill_name}-agent"

    # ── 3. Provider info for overview ───────────────────────────────────
    provider_name = config.get("providers", {}).get("default", "openai")
    provider_cfg = config.get("providers", {}).get(provider_name, {})
    if isinstance(provider_cfg, dict):
        model_display = provider_cfg.get("default_model", "unknown")
    else:
        model_display = "unknown"

    # ── 4. Early output dir conflict check (fail fast, before LLM cost) ─
    if (
        not force
        and not no_generate
        and not dry_run
        and agent_output_dir_early.exists()
    ):
        console.print()
        console.print(Panel(
            f"Output directory already exists:\n"
            f"[dim]{agent_output_dir_early}[/dim]\n\n"
            f"Use [bold]--force[/bold] to overwrite, "
            f"or [bold]--output[/bold] to choose a different path.",
            title="[error]Directory Conflict[/error]",
            border_style="red",
        ))
        raise typer.Exit(code=2)

    # ── 5. Overview panel ───────────────────────────────────────────────
    console.print()
    home = Path.home()
    dest_display = str(agent_output_dir_early)
    if dest_display.startswith(str(home)):
        dest_display = "~" + dest_display[len(str(home)):]

    console.print(Panel(
        f"[bold]{skill_name}[/bold] → [dim]{dest_display}[/dim]\n"
        f"[dim]{provider_name} / {model_display}[/dim]",
        title="[accent]agenthatch hatch[/accent]",
        border_style="cyan",
    ))

    # ── 6. Phase 1: Context Assembly ────────────────────────────────────
    console.print("[accent]▸ Phase 1/3[/accent]  Context Assembly")
    t1 = time.time()

    try:
        context = assemble_context(skill_real_path)
    except FileNotFoundError as e:
        console.print(f"[error]Error: {e}[/error]")
        raise typer.Exit(code=1) from e
    except Exception as e:
        console.print(f"[error]Parse error: {e}[/error]")
        raise typer.Exit(code=3) from e

    elapsed1 = time.time() - t1
    total_files = len(context.file_manifest.entries)
    total_size = sum(e.size_bytes for e in context.file_manifest.entries)
    size_str = _format_size(total_size)
    console.print(
        f"[ok]✓[/ok]  {total_files} files indexed · {size_str} · {elapsed1:.1f} seconds"
    )

    if trace:
        for entry in context.file_manifest.entries[:20]:
            console.print(f"    [dim]{entry.path}[/dim]")
        if total_files > 20:
            console.print(f"    [dim]... and {total_files - 20} more[/dim]")
        for warning in context.parse_warnings:
            console.print(f"    [warn]⚠ {warning}[/warn]")

    # ── 7. Phase 2: Agentic Inference ───────────────────────────────────
    console.print("[accent]▸ Phase 2/3[/accent]  Agentic Inference")
    t2 = time.time()

    harness_cfg = config.get("harness", {})
    large_model = harness_cfg.get("large_model", "") if isinstance(harness_cfg, dict) else ""
    small_model = harness_cfg.get("small_model", "") if isinstance(harness_cfg, dict) else ""

    try:
        with console.status(
            "[dim]Analyzing skill structure (Harness A→B→C→D→E)...[/dim]",
            spinner="dots",
        ):
            ahs_spec, harness_outputs = build_ahspec(
                context, config, large_model=large_model, small_model=small_model
            )
    except Exception as e:
        console.print(f"[error]Inference error: {e}[/error]")
        raise typer.Exit(code=4) from e

    elapsed2 = time.time() - t2
    harness_count = len(harness_outputs)
    console.print(
        f"[ok]✓[/ok]  {harness_count} harnesses completed · {elapsed2:.1f} seconds"
    )

    if trace:
        _render_harness_traces(harness_outputs)

    # ── 7.5. Phase 2.5: Skill Classification (v0.7) ─────────────────
    from agenthatch_core.bricks.archetypes import classify_skill

    classification = classify_skill(ahs_spec.model_dump() if hasattr(ahs_spec, "model_dump") else ahs_spec)
    console.print(
        f"     [dim]Archetype: {classification.archetype.value} "
        f"(confidence: {classification.confidence:.0%})[/dim]"
    )

    # ── 8. Confidence panel ─────────────────────────────────────────────
    # ── 9. Dry-run YAML/JSON output ─────────────────────────────────────
    if dry_run:
        console.print()
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

    # ── 10. Ensure v0.6 agent status ────────────────────────────────────
    if ahs_spec.agent is None:
        ahs_spec.agent = AgentConfig(status="not_generated")
    else:
        ahs_spec.agent.status = "not_generated"

    # ── 11. Write agenthatch.yaml ───────────────────────────────────────
    if not dry_run:
        yaml_output_path = _resolve_yaml_path(skill_dir, output)
        if yaml_output_path.exists() and not force:
            console.print(
                f"[dim]agenthatch.yaml already exists at {yaml_output_path}, "
                "skipping yaml generation.[/dim]"
            )
        else:
            yaml_output_path.parent.mkdir(parents=True, exist_ok=True)
            yaml_str = yaml.dump(
                json.loads(
                    ahs_spec.model_dump_json(
                        exclude={"harness_traces", "confidence_report"}
                    )
                ),
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            )
            yaml_output_path.write_text(yaml_str, encoding="utf-8")
            console.print(f"  [dim]Written: {yaml_output_path}[/dim]")

        # ── Register in skillhouse.json ─────────────────────────────────
        _register_skillhouse(ahs_spec, yaml_output_path, config)

    # ── 12. Phase 3: Agent Generation ───────────────────────────────────
    if no_generate:
        console.print(
            "[accent]▸ Phase 3/3[/accent]  Agent Generation  [dim](skipped)[/dim]"
        )
        return

    if dry_run:
        console.print(
            "[accent]▸ Phase 3/3[/accent]  Agent Generation  [dim](dry-run)[/dim]"
        )
    else:
        console.print("[accent]▸ Phase 3/3[/accent]  Agent Generation")

    t3 = time.time()
    file_count, agent_output_dir = _run_phase3_generate(
        ahs_spec=ahs_spec,
        skill_dir=skill_dir,
        output=output,
        force=force,
        dry_run=dry_run,
        copy_skills=not no_copy_skills,
        _framework=framework,
    )
    elapsed3 = time.time() - t3

    _render_phase3_result(
        file_count=file_count,
        agent_output_dir=agent_output_dir,
        elapsed=elapsed3,
        dry_run=dry_run,
        trace=trace,
        written_files=list(agent_output_dir.glob("**/*")) if trace and not dry_run else None,
    )

    # ── Update skillhouse index with agent output path ──────────────────
    if not dry_run:
        _update_skillhouse_agent_output(ahs_spec.identity.id, agent_output_dir, config)

    # ── 13. Confidence panel ─────────────────────────────────────────────
    _render_confidence(ahs_spec)

    # ── 14. Next step ───────────────────────────────────────────────────
    if not no_generate and not dry_run:
        console.print(
            f"[dim]Next step:[/dim] [bold]agenthatch run {ahs_spec.identity.id}[/bold]"
        )


# ── Internal helpers (yaml, skillhouse, output path) ────────────────────────


def _resolve_yaml_path(skill_dir: Path, output: str | None) -> Path:
    """Resolve where to write agenthatch.yaml."""
    if output:
        agent_output_dir = Path(output).expanduser().resolve()
        return agent_output_dir / "agenthatch.yaml"
    return skill_dir / "agenthatch.yaml"


def _register_skillhouse(
    ahs_spec: Any, yaml_output_path: Path, config: dict[str, Any]
) -> None:
    """Register skill in skillhouse.json index."""
    skillhouse_config = config.get("skillhouse", {}) if "skillhouse" in config else {}
    skillhouse_path = skillhouse_config.get(
        "path", ".agenthatch/skillhouse.json"
    ) if isinstance(skillhouse_config, dict) else ".agenthatch/skillhouse.json"

    skillhouse_full_path = Path(skillhouse_path)
    if not skillhouse_full_path.is_absolute():
        skillhouse_full_path = Path.cwd() / skillhouse_full_path

    from agenthatch.house.index import SkillhouseIndex

    idx = SkillhouseIndex(str(skillhouse_full_path))
    try:
        idx.add_entry(ahs_spec.identity.id, ahs_spec, str(yaml_output_path))
        console.print(
            f"  [dim]Registered: {skillhouse_full_path} "
            f"({idx.entry_count} entries)[/dim]"
        )
    except Exception as e:
        logger.warning(
            f"Hatch succeeded but skillhouse index update failed: {e}. "
            f"The agenthatch.yaml is valid and can be used."
        )
        console.print(f"[yellow]⚠ Skill indexed failed (non-fatal): {e}[/yellow]")


def _update_skillhouse_agent_output(
    agent_id: str, agent_output_dir: Path, config: dict[str, Any]
) -> None:
    """Update skillhouse index with agent output path (non-fatal)."""
    skillhouse_config = config.get("skillhouse", {}) if "skillhouse" in config else {}
    skillhouse_path = skillhouse_config.get(
        "path", ".agenthatch/skillhouse.json"
    ) if isinstance(skillhouse_config, dict) else ".agenthatch/skillhouse.json"

    skillhouse_full_path = Path(skillhouse_path)
    if not skillhouse_full_path.is_absolute():
        skillhouse_full_path = Path.cwd() / skillhouse_full_path

    if not skillhouse_full_path.exists():
        return

    from agenthatch.house.index import SkillhouseIndex

    try:
        idx = SkillhouseIndex(str(skillhouse_full_path))
        idx.update_agent_output(agent_id, str(agent_output_dir))
    except Exception:
        pass


# ── Name Resolution (3-layer fallback) ─────────────────────────────────────


def _resolve_skill_name(skill_input: str, config: dict[str, Any]) -> tuple[Path, bool]:
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


def _resolve_skill_dir(path: Path) -> Path:
    """Resolve a path to its skill directory."""
    if path.is_file() and path.suffix in (".md", ".markdown"):
        return path.parent
    return path


def _resolve_from_index(name: str, config: dict[str, Any]) -> Path | None:
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


def _resolve_from_filesystem(name: str, config: dict[str, Any]) -> Path:
    """BFS scan search_dirs for a directory named 'name' containing SKILL.md.

    Pattern: BFS with depth/dir limits, case-insensitive SKILL.md matching.

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

    Pattern: BFS with deque, depth limit, dir count limit, skip dot-prefixed dirs,
    case-insensitive SKILL.md matching.

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

        if has_skill_md and current_dir.name == target_name:
            results.append(current_dir)

        for subdir in subdirs:
            visited.add(subdir)
            queue.append((subdir, depth + 1))

    return results


def _auto_register_to_index(skill_dir: Path, config: dict[str, Any]) -> None:
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
        pass


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

_KNOWN_SKILL_HOST_DIRS: list[str] = [
    ".claude/skills",
    ".openclaw/skills",
    ".codex/skills",
    ".agents/skills",
    "skills",
    ".agenthatch/skills",
]


def _resolve_search_roots(config: dict[str, Any]) -> list[Path]:
    """Resolve all skill search roots from three sources.

    Sources:
      1. [skills].search_dirs from config (user-specified)
      2. _KNOWN_SKILL_HOST_DIRS under $HOME (auto-discovered AI tool dirs)
      3. Project-level .agents/skills/ (convention, up to 3 parent levels)
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

    # Source 3: Project-level .agents/skills/ (convention)
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
