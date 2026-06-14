# ruff: noqa: E501
#!/usr/bin/env python3
"""Batch hatch + chat test for 20 skills — open-source readiness validation.

Uses agenthatch CLI directly for the most realistic test.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).parent.parent  # .../project/agenthatch/
DEVELOPER = Path(__file__).parent.parent.parent.parent  # .../agenthatch_developer/
PYTHON = "/opt/homebrew/bin/python3.14"
OUTPUT_DIR = Path("/tmp/agenthatch_20_test")

# ─── Collect all 20 skills ────────────────────────────────────────────

SKILLS: dict[str, Path] = {}

# Anthropic skills (from test_skills/anthropic/)
ANTHROPIC_DIR = DEVELOPER / "test_skills" / "anthropic"
for d in sorted(ANTHROPIC_DIR.iterdir()):
    if d.is_dir() and not d.name.endswith("-agent"):
        md = d / "SKILL.md"
        if md.exists():
            SKILLS[d.name] = md

# Community skills (from test_skills/community/)
COMMUNITY_DIR = DEVELOPER / "test_skills" / "community"
for d in sorted(COMMUNITY_DIR.iterdir()):
    if d.is_dir():
        md = d / "SKILL.md"
        if md.exists():
            SKILLS[d.name] = md

# Fixture skills
FIXTURE_DIR = PROJECT / "tests" / "fixtures" / "skills"
for d in sorted(FIXTURE_DIR.iterdir()):
    if d.is_dir():
        md = d / "SKILL.md"
        if md.exists():
            SKILLS[d.name] = md

# Project skill
sha256_md = PROJECT / "sha256-checker" / "skills" / "SKILL.md"
if sha256_md.exists():
    SKILLS["sha256-checker"] = sha256_md


def run_hatch(name: str, skill_path: Path) -> dict:
    """Run agenthatch hatch for a single skill."""
    t0 = time.time()
    try:
        result = subprocess.run(
            [PYTHON, "-m", "agenthatch", "hatch", str(skill_path),
             "--output", str(OUTPUT_DIR / name),
             "--force"],
            cwd=str(PROJECT),
            capture_output=True,
            text=True,
            timeout=300,  # 5 min per hatch
            env={**os.environ, "PYTHONPATH": f"{PROJECT}/src:{PROJECT}/agenthatch-core/src"},
        )
        duration = time.time() - t0
        return {
            "pass": result.returncode == 0,
            "duration": duration,
            "stdout": result.stdout[-2000:] if result.stdout else "",
            "stderr": result.stderr[-1000:] if result.stderr else "",
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"pass": False, "duration": 300, "error": "TIMEOUT (5min)"}
    except Exception as e:
        return {"pass": False, "duration": time.time() - t0, "error": str(e)}


def run_chat(name: str) -> dict:
    """Test chat with a hatched agent."""
    t0 = time.time()
    ahs_path = OUTPUT_DIR / name / "agenthatch.yaml"
    if not ahs_path.exists():
        return {"pass": False, "duration": 0, "error": "No agenthatch.yaml"}

    chat_script = str(PROJECT / "tests" / "chat_test.py")
    try:
        result = subprocess.run(
            [PYTHON, chat_script, name, str(OUTPUT_DIR)],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(PROJECT),
            env={**os.environ, "PYTHONPATH": f"{PROJECT}/src:{PROJECT}/agenthatch-core/src"},
        )
        duration = time.time() - t0
        response = result.stdout.strip()
        return {
            "pass": len(response) > 20 and result.returncode == 0,
            "duration": duration,
            "response": response[:300],
            "error": result.stderr[:500] if result.returncode != 0 else "",
        }
    except subprocess.TimeoutExpired:
        return {"pass": False, "duration": 60, "error": "TIMEOUT"}
    except Exception as e:
        return {"pass": False, "duration": time.time() - t0, "error": str(e)}


def main():
    print(f"agenthatch Batch Test — {len(SKILLS)} skills")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Skills: {list(SKILLS.keys())}")
    print()

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    results = {}
    passed = 0
    failed = 0

    for i, (name, skill_path) in enumerate(SKILLS.items()):
        print(f"[{i+1:2d}/{len(SKILLS)}] {name} ", end="", flush=True)

        hr = run_hatch(name, skill_path)
        if hr["pass"]:
            cr = run_chat(name)
            if cr["pass"]:
                print(f"HATCH + CHAT PASS ({hr['duration']:.0f}s + {cr['duration']:.0f}s)")
                passed += 1
            else:
                print(f"HATCH PASS, CHAT FAIL: {cr.get('error', cr.get('response', '?'))[:80]}")
                failed += 1
        else:
            print(f"HATCH FAIL: {hr.get('error', hr.get('stderr', '?'))[:80]}")
            failed += 1

        results[name] = {"hatch": hr, "chat": cr if hr["pass"] else None}

    # Save report
    report = {
        "total": len(SKILLS),
        "passed": passed,
        "failed": failed,
        "results": {
            name: {
                "hatch_pass": r["hatch"]["pass"],
                "hatch_duration": r["hatch"]["duration"],
                "chat_pass": r["chat"]["pass"] if r["chat"] else False,
                "chat_duration": r["chat"]["duration"] if r["chat"] else 0,
                "chat_response": r["chat"]["response"][:200] if r["chat"] and r["chat"]["pass"] else "",
            }
            for name, r in results.items()
        },
    }
    report_path = OUTPUT_DIR / "batch_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"\n{'='*60}")
    print(f"RESULT: {passed}/{len(SKILLS)} passed, {failed} failed")
    print(f"Report: {report_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
