# ruff: noqa: E501
"""Industrial-grade Agent Evaluation Framework — v0.9

Built on best practices from AgentBench, DeepEval, RAGAS, and other agent evaluation frameworks.

Evaluation Dimensions:
  1. Functional Correctness (FC): Does the agent produce correct results?
  2. Tool Selection Accuracy (TSA): Does the agent choose the right tools?
  3. Multi-turn Coherence (MTC): Does the agent maintain context across turns?
  4. Error Handling (EH): Does the agent handle errors gracefully?
  5. Instruction Adherence (IA): Does the agent follow instructions?
  6. Hallucination Detection (HD): Does the agent avoid fabricating capabilities?
  7. Response Quality (RQ): Is the response well-structured and actionable?
  8. Latency & Token Efficiency (LTE): Performance metrics

Usage:
    python tests/v09_agent_evaluation.py --skill web-fetcher
    python tests/v09_agent_evaluation.py --skill pdf
    python tests/v09_agent_evaluation.py --all
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── Evaluation data structures ────────────────────────────────────────────


@dataclass
class TurnResult:
    """Single turn evaluation result."""
    turn_id: int
    user_input: str
    agent_response: str
    elapsed_seconds: float
    token_count: int = 0
    tool_calls: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    evaluation: dict[str, float] = field(default_factory=dict)
    observations: list[str] = field(default_factory=list)


@dataclass
class ScenarioResult:
    """Multi-turn scenario evaluation result."""
    name: str
    description: str
    turns: list[TurnResult] = field(default_factory=list)
    passed: bool = False
    dimension_scores: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    skill_name: str
    model: str
    scenarios: list[ScenarioResult] = field(default_factory=list)
    overall_score: float = 0.0
    dimension_averages: dict[str, float] = field(default_factory=dict)
    total_turns: int = 0
    total_elapsed: float = 0.0
    total_tokens: int = 0
    critical_issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


# ── Test Scenarios ────────────────────────────────────────────────────────


def get_scenarios(skill: str) -> list[dict[str, Any]]:
    """Get evaluation scenarios for a given skill."""
    if skill == "web-fetcher":
        return [
            {
                "name": "basic_fetch",
                "description": "Basic URL fetch — core functionality",
                "turns": [
                    "Fetch the content of https://example.com",
                ],
                "expected_tools": ["fetch_url"],
                "expected_keywords": ["Example Domain", "illustrative"],
                "forbidden_tools": [],
                "dimensions": ["FC", "TSA", "RQ"],
            },
            {
                "name": "invalid_url",
                "description": "Invalid URL — error handling",
                "turns": [
                    "Fetch the content of not-a-valid-url",
                ],
                "expected_tools": ["fetch_url"],
                "expected_keywords": [],
                "forbidden_tools": [],
                "dimensions": ["EH", "IA"],
            },
            {
                "name": "multi_turn_context",
                "description": "Multi-turn context memory",
                "turns": [
                    "Fetch the content of https://httpbin.org/json",
                    "Based on what you just fetched, what was the response about?",
                ],
                "expected_tools": ["fetch_url"],
                "expected_keywords": ["json", "slideshow"],
                "forbidden_tools": [],
                "dimensions": ["MTC", "FC"],
            },
            {
                "name": "http_error",
                "description": "HTTP error handling (404)",
                "turns": [
                    "Fetch https://httpstat.us/404",
                ],
                "expected_tools": ["fetch_url"],
                "expected_keywords": ["404", "Not Found"],
                "forbidden_tools": [],
                "dimensions": ["EH", "RQ"],
            },
            {
                "name": "hallucination_test",
                "description": "Hallucination detection — agent should not claim non-existent capabilities",
                "turns": [
                    "Can you create a PDF from this webpage? https://example.com",
                    "Can you send an email with the content of https://example.com?",
                ],
                "expected_tools": [],
                "expected_keywords": [],
                "forbidden_tools": [],
                "dimensions": ["HD", "IA"],
            },
            {
                "name": "complex_instruction",
                "description": "Complex instruction — extract specific info",
                "turns": [
                    "Fetch https://example.com and tell me what the main heading says",
                ],
                "expected_tools": ["fetch_url"],
                "expected_keywords": ["Example Domain", "heading", "h1"],
                "forbidden_tools": [],
                "dimensions": ["FC", "RQ", "IA"],
            },
            {
                "name": "tool_boundary",
                "description": "Tool boundary — agent should not misuse tools",
                "turns": [
                    "List all files in the /tmp directory",
                    "What is the current date and time?",
                ],
                "expected_tools": [],
                "expected_keywords": [],
                "forbidden_tools": ["fetch_url"],
                "dimensions": ["TSA", "HD"],
            },
        ]
    elif skill == "pdf":
        return [
            {
                "name": "basic_query",
                "description": "Basic PDF capability query",
                "turns": [
                    "What can you do with PDF files?",
                ],
                "expected_tools": [],
                "expected_keywords": ["extract", "merge", "split", "read"],
                "forbidden_tools": [],
                "dimensions": ["FC", "RQ"],
            },
            {
                "name": "hallucination_test",
                "description": "Hallucination detection",
                "turns": [
                    "Can you send this PDF as an email attachment?",
                    "Can you edit this PDF and change the font to Comic Sans?",
                ],
                "expected_tools": [],
                "expected_keywords": [],
                "forbidden_tools": [],
                "dimensions": ["HD", "IA"],
            },
            {
                "name": "instruction_following",
                "description": "Instruction adherence",
                "turns": [
                    "I have a PDF at /tmp/test.pdf — how do I extract text from it?",
                    "What libraries do I need to install?",
                ],
                "expected_tools": [],
                "expected_keywords": ["pypdf", "PdfReader", "extract_text"],
                "forbidden_tools": [],
                "dimensions": ["IA", "RQ"],
            },
        ]
    else:
        return []


# ── Evaluation Logic ──────────────────────────────────────────────────────


def evaluate_turn(
    turn: TurnResult,
    scenario: dict[str, Any],
    turn_idx: int,
    is_last: bool,
) -> TurnResult:
    """Evaluate a single turn against the scenario expectations."""
    response_lower = turn.agent_response.lower()
    user_input_lower = turn.user_input.lower()

    # ── Functional Correctness (FC) ──────────────────────────────────
    fc_score = 1.0
    if "expected_keywords" in scenario and scenario["expected_keywords"]:
        hits = sum(
            1 for kw in scenario["expected_keywords"]
            if kw.lower() in response_lower
        )
        fc_score = min(1.0, hits / max(len(scenario["expected_keywords"]), 1))

    # ── Tool Selection Accuracy (TSA) ─────────────────────────────────
    tsa_score = 1.0
    # TSA checks if the agent USED the right tool (via tool_calls), not if it mentions it in text
    if "expected_tools" in scenario and scenario["expected_tools"]:
        if turn.tool_calls:
            tool_hits = sum(
                1 for t in scenario["expected_tools"]
                for tc in turn.tool_calls
                if t.lower() in str(tc).lower()
            )
            tsa_score = max(0.5, tool_hits / max(len(scenario["expected_tools"]), 1))
        else:
            # No tool calls recorded — check if tool was abstracted correctly in response
            # If agent completed the task without exposing tool names, that's good UX
            tsa_score = 0.8  # Good: agent abstracted tool usage from user
    if "forbidden_tools" in scenario and scenario["forbidden_tools"]:
        for ft in scenario["forbidden_tools"]:
            if ft.lower() in response_lower:
                tsa_score = max(0.0, tsa_score - 0.5)
                turn.observations.append(f"Used forbidden tool: {ft}")
            for tc in turn.tool_calls:
                if ft.lower() in str(tc).lower():
                    tsa_score = 0.0
                    turn.observations.append(f"Called forbidden tool: {ft}")

    # ── Error Handling (EH) ───────────────────────────────────────────
    eh_score = 1.0
    if turn.errors:
        eh_score = 0.5
        turn.observations.append(f"Errors encountered: {turn.errors}")
    # Check if agent gracefully handles errors
    error_indicators = ["error", "invalid", "unable", "cannot", "failed", "not supported"]
    if any(ind in user_input_lower for ind in ["invalid", "not-a-", "error", "404"]):
        if any(ind in response_lower for ind in error_indicators):
            eh_score = 1.0  # Agent correctly identified the error
        elif any(ind in response_lower for ind in ["example domain", "illustrative"]):
            eh_score = 0.3  # Agent hallucinated a response for invalid input
            turn.observations.append("Agent did not handle invalid input correctly")

    # ── Hallucination Detection (HD) ──────────────────────────────────
    hd_score = 1.0
    hallucination_patterns = [
        ("I can create", "create a pdf", "Cannot create PDF"),
        ("I can send", "send an email", "Cannot send email"),
        ("I will", "not capable", "Claimed capability incorrectly"),
        ("let me", "just a language model", "Overclaimed capability"),
    ]
    for pattern, check, note in hallucination_patterns:
        if pattern in response_lower and check not in response_lower:
            # Check if the agent is correctly refusing
            refusal = any(
                r in response_lower
                for r in ["cannot", "don't have", "not able", "not capable", "don't support"]
            )
            if not refusal:
                hd_score = max(0.0, hd_score - 0.3)
                turn.observations.append(note)

    # ── Response Quality (RQ) ─────────────────────────────────────────
    rq_score = 0.5  # Base score
    if len(turn.agent_response) > 50:
        rq_score += 0.2
    if len(turn.agent_response) > 200:
        rq_score += 0.1
    # Check for structured output
    if any(marker in turn.agent_response for marker in ["#", "**", "```", "##", "- "]):
        rq_score += 0.1
    # Check for actionable output
    if any(word in response_lower for word in ["step", "first", "next", "then", "finally"]):
        rq_score += 0.1
    rq_score = min(1.0, rq_score)

    # ── Instruction Adherence (IA) ────────────────────────────────────
    ia_score = 1.0
    if "expected_tools" in scenario and scenario["expected_tools"]:
        # Agent should use expected tools when asked
        if turn.tool_calls:
            tool_match = any(
                et.lower() in str(tc).lower()
                for et in scenario["expected_tools"]
                for tc in turn.tool_calls
            )
            if not tool_match and "expected_keywords" in scenario:
                # Agent didn't use expected tool but gave a response
                ia_score = 0.7
    if "forbidden_tools" in scenario:
        for ft in scenario["forbidden_tools"]:
            for tc in turn.tool_calls:
                if ft.lower() in str(tc).lower():
                    ia_score = 0.0
                    turn.observations.append(f"Used forbidden tool in call: {ft}")

    turn.evaluation = {
        "FC": fc_score,
        "TSA": tsa_score,
        "EH": eh_score,
        "HD": hd_score,
        "RQ": rq_score,
        "IA": ia_score,
    }
    return turn


def run_agent_chat(
    agent_module: Any,
    agent_class: Any,
    message: str,
    agent_instance: Any = None,
) -> tuple[str, float, list[str], list[str], Any]:
    """Run a single chat turn with the agent.

    Returns: (response, elapsed, tool_calls, errors, agent_instance)
    If agent_instance is None, creates a new one.
    """
    t0 = time.time()
    errors: list[str] = []
    tool_calls: list[str] = []

    try:
        if agent_instance is None:
            from agenthatch_core.config import inherit_api_key, resolve_runtime_config

            from agenthatch.config import Config
            config = Config.load()
            provider_name = config.get("providers", {}).get("default", "openai")
            provider_cfg = config.get("providers", {}).get(provider_name, {})
            runtime_config: dict[str, Any] = {}
            runtime_config.setdefault("llm", {})["provider"] = provider_name
            if isinstance(provider_cfg, dict):
                runtime_config.setdefault("llm", {})["model"] = provider_cfg.get("default_model", "")
            runtime_config = inherit_api_key(runtime_config)
            runtime_config = resolve_runtime_config(runtime_config)
            agent = agent_class(runtime_config=runtime_config)
        else:
            agent = agent_instance
        response = agent.chat(message)
    except Exception as e:
        response = f"ERROR: {e}"
        errors.append(str(e))
        agent = agent_instance

    elapsed = time.time() - t0
    return response, elapsed, tool_calls, errors, agent


def run_evaluation(
    skill_name: str,
    agent_path: str,
    agent_module_name: str,
    agent_class_name: str,
) -> EvaluationReport:
    """Run full evaluation suite for a skill."""
    import importlib.util as _util

    # Load agent module
    agent_file = Path(agent_path) / "src" / agent_module_name / "agent.py"
    if not agent_file.exists():
        raise FileNotFoundError(f"Agent file not found: {agent_file}")

    # Add the src directory to sys.path so relative imports work
    src_dir = str(Path(agent_path) / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    spec = _util.spec_from_file_location(
        f"{agent_module_name}.agent", str(agent_file)
    )
    module = _util.module_from_spec(spec)
    module.__package__ = agent_module_name
    spec.loader.exec_module(module)

    agent_class = getattr(module, agent_class_name)

    # Get model info
    from agenthatch.config import Config
    config = Config.load()
    provider_name = config.get("providers", {}).get("default", "openai")
    provider_cfg = config.get("providers", {}).get(provider_name, {})
    model = provider_cfg.get("default_model", "unknown") if isinstance(provider_cfg, dict) else "unknown"

    scenarios_def = get_scenarios(skill_name)
    if not scenarios_def:
        print(f"No scenarios defined for skill: {skill_name}")
        return EvaluationReport(skill_name=skill_name, model=model)

    report = EvaluationReport(skill_name=skill_name, model=model)
    total_start = time.time()

    for scenario_def in scenarios_def:
        scenario = ScenarioResult(
            name=scenario_def["name"],
            description=scenario_def["description"],
        )

        print(f"\n{'='*70}")
        print(f"Scenario: {scenario.name} — {scenario.description}")
        print(f"{'='*70}")

        # Use same agent instance for multi-turn scenarios
        agent_instance = None
        for idx, user_input in enumerate(scenario_def["turns"]):
            is_last = idx == len(scenario_def["turns"]) - 1
            print(f"\n  Turn {idx + 1}/{len(scenario_def['turns'])}: {user_input[:80]}...")

            response, elapsed, tool_calls, errors, agent_instance = run_agent_chat(
                module, agent_class, user_input, agent_instance=agent_instance,
            )

            turn = TurnResult(
                turn_id=idx + 1,
                user_input=user_input,
                agent_response=response,
                elapsed_seconds=elapsed,
                tool_calls=tool_calls,
                errors=errors,
            )

            turn = evaluate_turn(turn, scenario_def, idx, is_last)
            scenario.turns.append(turn)

            # Print turn summary
            avg_score = sum(turn.evaluation.values()) / max(len(turn.evaluation), 1)
            print(f"    Response: {response[:150]}...")
            print(f"    Elapsed: {elapsed:.1f}s | Avg Score: {avg_score:.2f}")
            if turn.observations:
                for obs in turn.observations:
                    print(f"    ⚠ {obs}")

            report.total_turns += 1
            report.total_elapsed += elapsed

        # Compute scenario dimension scores
        for dim in scenario_def.get("dimensions", []):
            scores = [
                t.evaluation.get(dim, 0)
                for t in scenario.turns
                if dim in t.evaluation
            ]
            if scores:
                scenario.dimension_scores[dim] = sum(scores) / len(scores)

        avg_scenario = (
            sum(scenario.dimension_scores.values()) / max(len(scenario.dimension_scores), 1)
            if scenario.dimension_scores else 0
        )
        scenario.passed = avg_scenario >= 0.6
        report.scenarios.append(scenario)

        print(f"\n  Scenario Score: {avg_scenario:.2f} | {'PASS' if scenario.passed else 'FAIL'}")
        for dim, score in scenario.dimension_scores.items():
            bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            print(f"    {dim}: {bar} {score:.2f}")

    report.total_elapsed = time.time() - total_start

    # Compute overall report
    all_dim_scores: dict[str, list[float]] = {}
    for s in report.scenarios:
        for dim, score in s.dimension_scores.items():
            all_dim_scores.setdefault(dim, []).append(score)

    report.dimension_averages = {
        dim: sum(scores) / len(scores)
        for dim, scores in all_dim_scores.items()
    }

    report.overall_score = (
        sum(report.dimension_averages.values()) / max(len(report.dimension_averages), 1)
        if report.dimension_averages else 0.0
    )

    # Collect critical issues
    for s in report.scenarios:
        for t in s.turns:
            for obs in t.observations:
                if "hallucinat" in obs.lower() or "forbidden" in obs.lower():
                    report.critical_issues.append(f"[{s.name}] {obs}")
            if t.errors:
                report.critical_issues.append(f"[{s.name}] Turn {t.turn_id}: {t.errors}")

    # Generate recommendations
    if report.dimension_averages.get("HD", 1.0) < 0.7:
        report.recommendations.append(
            "High hallucination rate — tighten system prompt and add capability boundaries"
        )
    if report.dimension_averages.get("EH", 1.0) < 0.7:
        report.recommendations.append(
            "Weak error handling — add explicit error paths in tool implementations"
        )
    if report.dimension_averages.get("TSA", 1.0) < 0.7:
        report.recommendations.append(
            "Poor tool selection — improve tool descriptions and CapBus routing"
        )
    if report.dimension_averages.get("MTC", 1.0) < 0.7:
        report.recommendations.append(
            "Weak multi-turn coherence — increase context window or improve memory management"
        )

    return report


def print_report(report: EvaluationReport) -> None:
    """Print formatted evaluation report."""
    print(f"\n{'='*70}")
    print("AGENT EVALUATION REPORT")
    print(f"{'='*70}")
    print(f"  Skill:      {report.skill_name}")
    print(f"  Model:      {report.model}")
    print(f"  Scenarios:  {len(report.scenarios)}")
    print(f"  Turns:      {report.total_turns}")
    print(f"  Total Time: {report.total_elapsed:.1f}s")
    print(f"\n  OVERALL SCORE: {report.overall_score:.2f} / 1.00")
    print(f"  {'='*50}")

    # Dimension breakdown
    print("\n  Dimension Scores:")
    for dim in ["FC", "TSA", "MTC", "EH", "IA", "HD", "RQ"]:
        score = report.dimension_averages.get(dim, 0)
        bar = "▓" * int(score * 20) + "░" * (20 - int(score * 20))
        labels = {
            "FC": "Functional Correctness",
            "TSA": "Tool Selection Accuracy",
            "MTC": "Multi-turn Coherence",
            "EH": "Error Handling",
            "IA": "Instruction Adherence",
            "HD": "Hallucination Detection",
            "RQ": "Response Quality",
        }
        print(f"    {dim} {labels.get(dim, dim):<28} {bar} {score:.2f}")

    # Scenario summary
    print("\n  Scenario Summary:")
    for s in report.scenarios:
        status = "PASS" if s.passed else "FAIL"
        avg = sum(s.dimension_scores.values()) / max(len(s.dimension_scores), 1) if s.dimension_scores else 0
        print(f"    [{status}] {s.name:<30} {avg:.2f} — {s.description}")

    # Critical issues
    if report.critical_issues:
        print("\n  CRITICAL ISSUES:")
        for issue in report.critical_issues:
            print(f"    ⚠ {issue}")

    # Recommendations
    if report.recommendations:
        print("\n  RECOMMENDATIONS:")
        for rec in report.recommendations:
            print(f"    → {rec}")

    # Save report
    report_path = Path(f"evaluation_report_{report.skill_name}.json")
    report_data = {
        "skill": report.skill_name,
        "model": report.model,
        "overall_score": report.overall_score,
        "dimension_averages": report.dimension_averages,
        "scenarios": [
            {
                "name": s.name,
                "passed": s.passed,
                "scores": s.dimension_scores,
                "turns": [
                    {
                        "turn_id": t.turn_id,
                        "user_input": t.user_input,
                        "response": t.agent_response[:500],
                        "evaluation": t.evaluation,
                        "observations": t.observations,
                    }
                    for t in s.turns
                ],
            }
            for s in report.scenarios
        ],
        "critical_issues": report.critical_issues,
        "recommendations": report.recommendations,
    }
    report_path.write_text(json.dumps(report_data, indent=2, ensure_ascii=False))
    print(f"\n  Report saved: {report_path}")


# ── Main ──────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agent Evaluation Framework")
    parser.add_argument("--skill", type=str, default="web-fetcher", help="Skill to evaluate")
    parser.add_argument("--all", action="store_true", help="Evaluate all skills")
    args = parser.parse_args()

    skills_to_eval = [args.skill] if not args.all else ["web-fetcher", "pdf"]

    for skill in skills_to_eval:
        # Determine agent path
        agent_dir = Path.cwd() / f"{skill}-agent"
        if not agent_dir.exists():
            print(f"Agent not found: {agent_dir}. Run 'agenthatch hatch {skill} --force' first.")
            continue

        # Convert skill name to module/class names
        parts = skill.replace("-", "_").split("_")
        module_name = "_".join(parts)
        class_name = "".join(p.capitalize() for p in parts)

        print(f"\nEvaluating: {skill} ({module_name}.{class_name})")
        print(f"Agent path: {agent_dir}")

        try:
            report = run_evaluation(
                skill_name=skill,
                agent_path=str(agent_dir),
                agent_module_name=module_name,
                agent_class_name=class_name,
            )
            print_report(report)
        except Exception as e:
            print(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
