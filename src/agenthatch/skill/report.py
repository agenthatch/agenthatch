"""HatchReport — Structured report for every hatch run (v0.9.17).

Aggregates Phase 1/2/3 telemetry into a single Pydantic model with dual
output: terminal (Rich) and JSON (CI-friendly).

Design constraints (v0.9.17):
- Reports NEVER block hatch. Verdict is PASS or WARN only — no FAIL.
- Existing progressive renderers (_render_confidence, _render_summary,
  _render_harness_traces, _render_phase3_result) are preserved; this
  report is an ADDITIONAL output triggered by --report.
- Token usage is sourced from HarnessOutput.token_usage, which is
  populated by AgentHarness.run() via _accumulate_token_usage().
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from agenthatch.skill.spec import HARNESS_LABELS

# ─────────────────────────────────────────────────────────────────────────
# Sub-models
# ─────────────────────────────────────────────────────────────────────────


class PhaseReport(BaseModel):
    """Telemetry for one hatch phase (1/2/3)."""

    name: str = Field(description="Phase identifier: 'phase_1_context', etc.")
    label: str = Field(description="Human-readable label")
    elapsed_seconds: float = 0.0
    token_usage: dict[str, int] = Field(default_factory=dict)
    status: Literal["ok", "skipped", "error"] = "ok"
    detail: str | None = None


class HarnessReport(BaseModel):
    """Telemetry for one AgentHarness (A–F)."""

    key: str = Field(description="Harness key: A, B, C, D, E, F")
    label: str = Field(description="Human-readable label, e.g. 'extract_identity'")
    confidence: float = 0.0
    self_check_passed: bool = True
    degradation_applied: list[str] = Field(default_factory=list)
    internal_retries: int = 0
    reasoning_trace: list[str] = Field(default_factory=list)
    token_usage: dict[str, int] = Field(default_factory=dict)


class ReadinessSummary(BaseModel):
    """Phase 4 readiness verdict (advisory only — never blocks)."""

    status: Literal["READY", "WARN"] = "READY"
    missing_optional: list[str] = Field(default_factory=list)
    fix_suggestions: list[str] = Field(default_factory=list)
    mcporter_installed: bool = False
    all_mcp_reachable: bool = True
    all_credentials_present: bool = True


# ─────────────────────────────────────────────────────────────────────────
# Top-level report
# ─────────────────────────────────────────────────────────────────────────


class HatchReport(BaseModel):
    """Structured hatch report — terminal + JSON dual output.

    v0.9.17: Verdict is PASS or WARN only. There is no FAIL state and
    the report never blocks hatch execution. The verdict is informational
    and intended for CI pipelines to surface quality signals without
    aborting the build.
    """

    skill_id: str
    skill_name: str
    generated_at: datetime
    provider: str | None = None
    model: str | None = None
    phases: list[PhaseReport] = Field(default_factory=list)
    harnesses: list[HarnessReport] = Field(default_factory=list)
    readiness: ReadinessSummary = Field(default_factory=ReadinessSummary)
    verdict: Literal["PASS", "WARN"] = "PASS"
    agent_output_dir: str | None = None
    file_count: int = 0
    archetype: str | None = None
    archetype_confidence: float | None = None
    total_tokens: dict[str, int] = Field(default_factory=dict)

    # ── Verdict computation ───────────────────────────────────────────

    def compute_verdict(self) -> Literal["PASS", "WARN"]:
        """Compute verdict from harness + readiness signals.

        Rules (v0.9.17 — no FAIL, no blocking):
        - WARN if any harness has degradation_applied
        - WARN if any harness self_check_passed is False
        - WARN if readiness.status == "WARN"
        - PASS otherwise
        """
        for h in self.harnesses:
            if h.degradation_applied:
                return "WARN"
            if not h.self_check_passed:
                return "WARN"
        if self.readiness.status == "WARN":
            return "WARN"
        return "PASS"

    # ── JSON output (CI-friendly) ─────────────────────────────────────

    def to_json(self) -> str:
        """Serialize to JSON string for CI consumption."""
        return self.model_dump_json(indent=2, exclude_none=True)

    # ── Terminal output (Rich) ────────────────────────────────────────

    def to_terminal(self) -> Group:
        """Build Rich renderables for terminal display.

        Returns a Group of:
        1. Header panel (skill + verdict)
        2. Phase timing table
        3. Harness detail table (confidence + self-check + tokens + degradations)
        4. Token summary
        5. Readiness section (if WARN)
        """
        # 1. Header
        verdict_style = "ok" if self.verdict == "PASS" else "warn"
        verdict_icon = "✓" if self.verdict == "PASS" else "⚠"
        header_lines = [
            f"[bold]Skill:[/bold]  {self.skill_name} [dim]({self.skill_id})[/dim]",
            f"[bold]Verdict:[/bold]  "
            f"[{verdict_style}]{verdict_icon} {self.verdict}[/{verdict_style}]",
        ]
        if self.provider and self.model:
            header_lines.append(f"[bold]Model:[/bold]  {self.provider} / {self.model}")
        if self.archetype:
            arch_conf = (
                f" ({self.archetype_confidence:.0%})"
                if self.archetype_confidence is not None
                else ""
            )
            header_lines.append(f"[bold]Archetype:[/bold]  {self.archetype}{arch_conf}")
        if self.agent_output_dir:
            header_lines.append(f"[bold]Output:[/bold]  {self.agent_output_dir}")
            header_lines.append(f"[bold]Files:[/bold]   {self.file_count}")
        header_lines.append(
            f"[bold]Generated:[/bold]  "
            f"{self.generated_at.isoformat(timespec='seconds')}"
        )

        header = Panel(
            "\n".join(header_lines),
            title="[accent]Hatch Report[/accent]",
            border_style="cyan",
        )

        renderables: list[Any] = [header]

        # 2. Phase timing table
        if self.phases:
            phase_table = Table(
                title="Phase Telemetry",
                border_style="dim",
                show_header=True,
                header_style="bold",
            )
            phase_table.add_column("Phase", style="accent")
            phase_table.add_column("Elapsed", justify="right")
            phase_table.add_column("Tokens", justify="right")
            phase_table.add_column("Status", justify="center")

            for p in self.phases:
                tok = p.token_usage.get("total_tokens", 0)
                tok_str = f"{tok:,}" if tok else "—"
                status_icon = {
                    "ok": "[ok]✓[/ok]",
                    "skipped": "[dim]⊘[/dim]",
                    "error": "[error]✗[/error]",
                }.get(p.status, p.status)
                phase_table.add_row(
                    p.label,
                    f"{p.elapsed_seconds:.2f}s",
                    tok_str,
                    status_icon,
                )

            # Total row
            total_elapsed = sum(p.elapsed_seconds for p in self.phases)
            total_tok = self.total_tokens.get("total_tokens", 0)
            phase_table.add_row(
                "[bold]Total[/bold]",
                f"[bold]{total_elapsed:.2f}s[/bold]",
                f"[bold]{total_tok:,}[/bold]" if total_tok else "[bold]—[/bold]",
                "",
            )
            renderables.append(phase_table)

        # 3. Harness detail table
        if self.harnesses:
            h_table = Table(
                title="Harness Detail",
                border_style="dim",
                show_header=True,
                header_style="bold",
            )
            h_table.add_column("Key", style="accent", justify="center")
            h_table.add_column("Task")
            h_table.add_column("Conf.", justify="right")
            h_table.add_column("Self-Check", justify="center")
            h_table.add_column("Retries", justify="right")
            h_table.add_column("Tokens", justify="right")
            h_table.add_column("Degradations")

            for h in self.harnesses:
                conf_str = f"{h.confidence:.2f}"
                check_str = "[ok]✓[/ok]" if h.self_check_passed else "[error]✗[/error]"
                tok = h.token_usage.get("total_tokens", 0)
                tok_str = f"{tok:,}" if tok else "—"
                deg_str = (
                    f"[warn]{len(h.degradation_applied)}[/warn]"
                    if h.degradation_applied
                    else "[dim]0[/dim]"
                )
                h_table.add_row(
                    h.key,
                    h.label,
                    conf_str,
                    check_str,
                    str(h.internal_retries),
                    tok_str,
                    deg_str,
                )
            renderables.append(h_table)

        # 4. Reasoning traces (compact, one tree per harness)
        if self.harnesses:
            trace_tree = Tree("[bold]Reasoning Traces[/bold]")
            for h in self.harnesses:
                branch = trace_tree.add(f"[bold]Harness {h.key}[/bold]: {h.label}")
                for line in h.reasoning_trace:
                    branch.add(f"[dim]{line}[/dim]")
                if h.degradation_applied:
                    branch.add(f"[warn]degradations: {h.degradation_applied}[/warn]")
            renderables.append(trace_tree)

        # 5. Token summary
        if self.total_tokens:
            tok_lines = [
                f"[bold]Prompt:[/bold]     {self.total_tokens.get('prompt_tokens', 0):,}",
                f"[bold]Completion:[/bold] {self.total_tokens.get('completion_tokens', 0):,}",
                f"[bold]Total:[/bold]     {self.total_tokens.get('total_tokens', 0):,}",
            ]
            renderables.append(
                Panel(
                    "\n".join(tok_lines),
                    title="[accent]Token Summary[/accent]",
                    border_style="dim",
                )
            )

        # 6. Readiness section
        if self.readiness.status == "WARN" or self.readiness.missing_optional:
            r_lines: list[str] = []
            r_lines.append(
                f"Status: [{'warn' if self.readiness.status == 'WARN' else 'ok'}]"
                f"{self.readiness.status}[/]"
            )
            if self.readiness.missing_optional:
                r_lines.append("")
                r_lines.append("[bold]Advisory warnings:[/bold]")
                for item in self.readiness.missing_optional:
                    r_lines.append(f"  • {item}")
            if self.readiness.fix_suggestions:
                r_lines.append("")
                r_lines.append("[bold]Fix suggestions:[/bold]")
                for s in self.readiness.fix_suggestions:
                    r_lines.append(f"  $ {s}")
            r_lines.append("")
            r_lines.append(
                "[dim]Note: agent will self-configure at runtime. "
                "No manual fix required to proceed.[/dim]"
            )
            renderables.append(
                Panel(
                    "\n".join(r_lines),
                    title="[accent]Readiness (Advisory)[/accent]",
                    border_style="yellow" if self.readiness.status == "WARN" else "dim",
                )
            )

        return Group(*renderables)


# ─────────────────────────────────────────────────────────────────────────
# Builder
# ─────────────────────────────────────────────────────────────────────────


def build_hatch_report(
    *,
    skill_id: str,
    skill_name: str,
    provider: str | None,
    model: str | None,
    phases: list[PhaseReport],
    harness_outputs: dict[str, Any],
    readiness: Any | None,
    agent_output_dir: str | None,
    file_count: int,
    archetype: str | None,
    archetype_confidence: float | None,
) -> HatchReport:
    """Construct a HatchReport from hatch telemetry.

    Args:
        skill_id: AHSSpec identity.id
        skill_name: AHSSpec identity.display_name
        provider: Provider name (e.g. "deepseek")
        model: Model name (e.g. "deepseek-v4-pro")
        phases: List of PhaseReport (phase 1/2/3 telemetry)
        harness_outputs: dict[str, HarnessOutput] from Orchestrator
        readiness: ReadinessVerdict from Phase 4, or None if skipped
        agent_output_dir: Final agent output directory, or None
        file_count: Number of files generated in Phase 3
        archetype: Skill archetype string, or None
        archetype_confidence: Archetype confidence 0.0-1.0, or None
    """
    # Build harness reports in canonical order A→B→C→D→E→F
    harness_reports: list[HarnessReport] = []
    total_tokens: dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    for key in ["A", "B", "C", "D", "E", "F"]:
        if key not in harness_outputs:
            continue
        h_out = harness_outputs[key]
        h_report = HarnessReport(
            key=key,
            label=HARNESS_LABELS.get(key, key),
            confidence=float(h_out.confidence),
            self_check_passed=bool(h_out.self_check_passed),
            degradation_applied=list(h_out.degradation_applied),
            internal_retries=int(h_out.internal_retries),
            reasoning_trace=list(h_out.reasoning_trace),
            token_usage=dict(h_out.token_usage),
        )
        harness_reports.append(h_report)

        # Accumulate harness tokens into total
        for k in total_tokens:
            total_tokens[k] += h_report.token_usage.get(k, 0)

    # Also accumulate phase tokens (Phase 3 AI tool generation)
    for p in phases:
        for k in total_tokens:
            total_tokens[k] += p.token_usage.get(k, 0)

    # Build readiness summary
    if readiness is not None:
        readiness_summary = ReadinessSummary(
            status=readiness.status if readiness.status in ("READY", "WARN") else "WARN",
            missing_optional=list(getattr(readiness, "missing_optional", [])),
            fix_suggestions=list(getattr(readiness, "fix_suggestions", [])),
            mcporter_installed=bool(getattr(readiness, "mcporter_installed", False)),
            all_mcp_reachable=bool(getattr(readiness, "all_mcp_reachable", True)),
            all_credentials_present=bool(
                getattr(readiness, "all_credentials_present", True)
            ),
        )
    else:
        readiness_summary = ReadinessSummary()

    report = HatchReport(
        skill_id=skill_id,
        skill_name=skill_name,
        generated_at=datetime.now(),
        provider=provider,
        model=model,
        phases=phases,
        harnesses=harness_reports,
        readiness=readiness_summary,
        agent_output_dir=agent_output_dir,
        file_count=file_count,
        archetype=archetype,
        archetype_confidence=archetype_confidence,
        total_tokens=total_tokens,
    )
    report.verdict = report.compute_verdict()
    return report
