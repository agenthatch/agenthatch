"""Phase 4 Runtime Readiness — v0.8 hatch-time environment verification.

Verifies that the generated agent can actually operate in the current
environment BEFORE the hatch is declared successful. Checks:
1. MCP CLI availability (mcporter)
2. Python package imports
3. System tool availability
4. MCP server reachability (network probe)
5. Credential presence
"""

from __future__ import annotations

import importlib
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("agenthatch")

# ─────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class DependencyManifest:
    """All external dependencies declared by a skill."""
    mcp_servers: list[dict[str, str]] = field(default_factory=list)
    # [{name: "cooper", transport: "streamable_http", url: "...", command: "..."}]
    pip_packages: list[str] = field(default_factory=list)
    # ["requests", "Pillow"]
    system_tools: list[str] = field(default_factory=list)
    # ["mcporter", "ffprobe", "git"]
    credential_keys: list[str] = field(default_factory=list)
    # ["cooper_token", "api_key"]


@dataclass
class EnvironmentReport:
    """What is actually available on this system."""
    mcporter: bool = False                  # shutil.which("mcporter") → found
    mcporter_version: str | None = None     # mcporter --version
    pip_packages: dict[str, bool] = field(default_factory=dict)
    # "requests" → True/False
    system_tools: dict[str, bool] = field(default_factory=dict)
    # "ffprobe" → True/False
    mcp_reachable: dict[str, bool] = field(default_factory=dict)
    # "cooper" → True/False
    credentials: dict[str, bool] = field(default_factory=dict)
    # "cooper_token" → True/False


@dataclass
class ReadinessVerdict:
    """Final ruling on whether the agent can operate."""
    status: str = "READY"                    # "READY" | "WARN" | "BLOCK"
    missing_mandatory: list[str] = field(default_factory=list)
    # Blocking: mcporter missing, required credential missing
    missing_optional: list[str] = field(default_factory=list)
    # Warning: optional dependency not found
    fix_suggestions: list[str] = field(default_factory=list)
    # Human-readable fix instructions
    mcporter_installed: bool = False
    all_mcp_reachable: bool = True
    all_credentials_present: bool = True


class HatchBlockedError(Exception):
    """Raised when hatch is blocked by missing mandatory dependencies."""
    pass


@dataclass
class HatchResult:
    """Complete hatch result including readiness status."""
    agent_path: str
    readiness: ReadinessVerdict = field(default_factory=ReadinessVerdict)
    report: str = ""
    compilability: bool = True
    tool_count: int = 0
    fidelity_score: float = 0.0


# ─────────────────────────────────────────────────────────────────────────
# Step 1: Dependency Manifest extraction
# ─────────────────────────────────────────────────────────────────────────

def extract_dependencies(
    skill_dir: Path,
    ahspec: dict[str, Any],
) -> DependencyManifest:
    """Extract all dependencies from skill analysis results.

    Sources:
    - AHSSPEC: MCP servers, base config
    - Script Manifest / Python files: import dependencies
    - SKILL.md: tool mentions
    """
    deps = DependencyManifest()

    # From AHSSPEC: MCP servers
    interface = ahspec.get("interface", {})
    mcp_servers = interface.get("mcp_servers", [])
    for mcp in mcp_servers:
        if isinstance(mcp, dict):
            deps.mcp_servers.append({
                "name": mcp.get("name", ""),
                "transport": mcp.get("transport", ""),
                "url": mcp.get("url", ""),
                "command": mcp.get("command", ""),
            })

    # From base config: detect credential references
    base = ahspec.get("base", {})
    env_vars = base.get("env", [])
    for env_var in env_vars:
        if isinstance(env_var, dict):
            name = env_var.get("name", "")
            required = env_var.get("required", False)
            if required and name:
                deps.credential_keys.append(name)
            elif name and (
                "_TOKEN" in name or "_KEY" in name or "_SECRET" in name
            ):
                deps.credential_keys.append(name)

    # From Python scripts: detect import dependencies
    scripts_dir = skill_dir / "skills" / "scripts"
    if scripts_dir.exists():
        deps.pip_packages = _detect_import_dependencies(scripts_dir)

    # System tools: mcporter is mandatory for MCP skills
    if deps.mcp_servers:
        if "mcporter" not in deps.system_tools:
            deps.system_tools.append("mcporter")

    return deps


def _detect_import_dependencies(scripts_dir: Path) -> list[str]:
    """Detect pip package dependencies from Python scripts via AST imports.

    Heuristic: imports that are NOT from stdlib or agenthatch_core
    are treated as external pip dependencies.
    """
    import ast as _ast
    import sys as _sys

    # Standard library module names (Python 3.11+)
    stdlib = set(_sys.stdlib_module_names) if hasattr(_sys, "stdlib_module_names") else set()

    found: set[str] = set()
    for py_file in scripts_dir.glob("*.py"):
        try:
            tree = _ast.parse(py_file.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue

        for node in _ast.walk(tree):
            if isinstance(node, _ast.Import):
                for alias in node.names:
                    name = alias.name.split(".")[0]
                    if name not in stdlib and not name.startswith("agenthatch"):
                        found.add(name)
            elif isinstance(node, _ast.ImportFrom):
                if node.module:
                    name = node.module.split(".")[0]
                    if name not in stdlib and not name.startswith("agenthatch"):
                        found.add(name)

    return sorted(found)


# ─────────────────────────────────────────────────────────────────────────
# Step 2: Environment Audit
# ─────────────────────────────────────────────────────────────────────────

def audit_environment(dep_manifest: DependencyManifest) -> EnvironmentReport:
    """Check what dependencies are actually available on this system."""
    report = EnvironmentReport()

    # 1. System tools: check PATH
    report.mcporter = shutil.which("mcporter") is not None
    if report.mcporter:
        report.mcporter_version = _get_cli_version("mcporter")

    for tool in dep_manifest.system_tools:
        report.system_tools[tool] = shutil.which(tool) is not None

    # 2. Python packages: check importability
    for pkg in dep_manifest.pip_packages:
        try:
            importlib.import_module(pkg)
            report.pip_packages[pkg] = True
        except ImportError:
            report.pip_packages[pkg] = False

    # 3. Credentials: check presence in runtime.toml
    for key in dep_manifest.credential_keys:
        report.credentials[key] = _check_credential_present(key)

    # 4. MCP servers: optional connectivity probe
    for mcp in dep_manifest.mcp_servers:
        report.mcp_reachable[mcp["name"]] = _probe_mcp_server(mcp)

    return report


def _get_cli_version(tool: str) -> str | None:
    """Get version string from a CLI tool."""
    try:
        result = subprocess.run(
            [tool, "--version"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip().split("\n")[0]
        return None
    except Exception:
        return None


def _check_credential_present(key: str) -> bool:
    """Check if a credential is configured.

    Checks common credential locations:
    1. Environment variable (direct key name)
    2. Environment variable (UPPER(key))
    3. runtime.toml [credentials] section (if found)
    """
    import os

    # Check environment variables
    if os.environ.get(key):
        return True
    if os.environ.get(key.upper()):
        return True

    # Check runtime.toml
    try:
        config_path = Path.cwd() / "runtime.toml"
        if config_path.exists():
            # Simple INI-style TOML check without full parser
            content = config_path.read_text(encoding="utf-8")
            if key in content:
                return True
    except Exception:
        pass

    return False


def _probe_mcp_server(mcp: dict[str, str]) -> bool:
    """Optional: probe MCP server reachability via mcporter."""
    if not shutil.which("mcporter"):
        return False  # Can't probe without mcporter
    try:
        result = subprocess.run(
            ["mcporter", "call", f"{mcp['name']}.list_tools"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False  # Network error, timeout, or other


# ─────────────────────────────────────────────────────────────────────────
# Step 3: Runtime Readiness Gate
# ─────────────────────────────────────────────────────────────────────────

def runtime_readiness_gate(
    dep_manifest: DependencyManifest,
    env_report: EnvironmentReport,
) -> ReadinessVerdict:
    """Determine if the agent can actually function in this environment.

    Classification rules:
    - BLOCK: mcporter missing AND skill uses MCP
    - BLOCK: mandatory credential missing
    - WARN: optional MCP server unreachable
    - WARN: optional pip package not installed
    - READY: all mandatory satisfied
    """
    verdict = ReadinessVerdict(status="READY")

    # Mandatory checks
    if dep_manifest.mcp_servers and not env_report.mcporter:
        verdict.status = "BLOCK"
        verdict.mcporter_installed = False
        verdict.missing_mandatory.append(
            "mcporter CLI not found. MCP tools will fail. "
            "Install with: npm install -g mcporter"
        )
        verdict.fix_suggestions.append(
            "npm install -g mcporter && mcporter config set token YOUR_TOKEN"
        )

    for key, present in env_report.credentials.items():
        if not present:
            verdict.all_credentials_present = False
            verdict.missing_mandatory.append(
                f"Credential '{key}' not configured in environment or runtime.toml. "
                f"Add it to the [credentials] section."
            )
            verdict.fix_suggestions.append(
                f"Add '{key} = \"your-value\"' to runtime.toml [credentials] section"
            )
            if verdict.status != "BLOCK":
                verdict.status = "WARN"

    # Optional checks (WARN only, not BLOCK)
    for tool, found in env_report.system_tools.items():
        if not found and tool != "mcporter":  # Already checked as mandatory
            verdict.missing_optional.append(
                f"System tool '{tool}' not found on PATH. "
                f"Install with your package manager."
            )

    for pkg, found in env_report.pip_packages.items():
        if not found:
            verdict.missing_optional.append(
                f"Python package '{pkg}' not installed. "
                f"Install with: pip install {pkg}"
            )
            verdict.fix_suggestions.append(f"pip install {pkg}")

    for mcp_name, reachable in env_report.mcp_reachable.items():
        if not reachable:
            verdict.all_mcp_reachable = False
            verdict.missing_optional.append(
                f"MCP server '{mcp_name}' is not reachable. "
                f"Check network, VPN, and server URL."
            )

    # Check if all mandatory requirements are met
    if not verdict.missing_mandatory and not verdict.missing_optional:
        verdict.status = "READY"
    elif verdict.missing_mandatory:
        verdict.status = "BLOCK"
    else:
        verdict.status = "WARN"

    return verdict


# ─────────────────────────────────────────────────────────────────────────
# Step 4: Hatch report formatting
# ─────────────────────────────────────────────────────────────────────────

def format_hatch_report(
    agent_path: str,
    verdict: ReadinessVerdict,
    ahspec: dict[str, Any],
) -> str:
    """Format a human-readable hatch report."""
    lines: list[str] = []

    identity = ahspec.get("identity", {})
    agent_name = identity.get("display_name", "Unknown")
    agent_id = identity.get("id", "unknown")

    lines.append(f"HATCH REPORT: {agent_id}")
    lines.append("")
    lines.append(f"  Agent:       {agent_name} ({agent_id})")
    lines.append(f"  Path:        {agent_path}")
    lines.append("")

    # Status
    status_icon = {"READY": "PASS", "WARN": "WARN", "BLOCK": "FAIL"}.get(
        verdict.status, "UNKNOWN"
    )
    lines.append(f"  Status:      {status_icon} {verdict.status}")
    lines.append("")

    # Missing mandatory
    if verdict.missing_mandatory:
        lines.append("  Blocking issues:")
        for item in verdict.missing_mandatory:
            lines.append(f"    - {item}")

    # Missing optional
    if verdict.missing_optional:
        lines.append("  Warnings:")
        for item in verdict.missing_optional:
            lines.append(f"    - {item}")

    # Fix suggestions
    if verdict.fix_suggestions:
        lines.append("")
        lines.append("  Fix suggestions:")
        for suggestion in verdict.fix_suggestions:
            lines.append(f"    $ {suggestion}")

    lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────
# Integration helper
# ─────────────────────────────────────────────────────────────────────────

def run_readiness_phase(
    skill_dir: Path,
    ahspec: dict[str, Any],
    agent_path: str,
    skip_network_probe: bool = False,
) -> HatchResult:
    """Run the complete Phase 4 readiness pipeline.

    Args:
        skill_dir: Path to the skill directory
        ahspec: Full AHSSPEC dict
        agent_path: Path where agent.py was written
        skip_network_probe: If True, skip MCP reachability probes

    Returns:
        HatchResult with readiness status and report
    """
    # Step 1: Extract dependencies
    dep_manifest = extract_dependencies(skill_dir, ahspec)

    # Step 2: Audit environment
    env_report = audit_environment(dep_manifest)

    # Step 3: Gate
    verdict = runtime_readiness_gate(dep_manifest, env_report)

    # Step 4: Report
    report = format_hatch_report(agent_path, verdict, ahspec)

    return HatchResult(
        agent_path=agent_path,
        readiness=verdict,
        report=report,
    )
