"""Real API key test: verify that a hatched SkillAgent can produce
substantive, high-quality responses — not just empty shell dialogue.

Uses LLMClient directly with the generated agent's system prompt.
"""
import sys

sys.path.insert(0, "/Users/didi/agenthatch_developer/project/agenthatch/src")
sys.path.insert(0, "/Users/didi/agenthatch_developer/project/agenthatch/agenthatch-core/src")

import re
from pathlib import Path

import yaml
from agenthatch_core.llm.client import LLMClient

from agenthatch.config import Config

# ── Load config ──
config = Config.load()

# ── Read generated AHSSPEC ──
ahspec_path = Path("/tmp/agenthatch_test/weather-reporter/agenthatch.yaml")
with open(ahspec_path) as f:
    ahspec = yaml.safe_load(f)

# ── Build system prompt from AHSSPEC (simulating what ContextManager does) ──
identity = ahspec["identity"]
instructions = ahspec["instructions"]
interface = ahspec["interface"]
resources = ahspec.get("resources", {})

system_prompt_parts = [
    f"You are {identity['display_name']}, v{identity['version']}.",
    f"Description: {ahspec.get('description', identity.get('description', 'An AI agent.'))}",
    "",
    "## Capabilities",
]

for cap in interface.get("provides", []):
    system_prompt_parts.append(f"- {cap['capability']}: ({cap.get('type', 'tool')}) {cap.get('description', '')}")  # noqa: E501

if instructions.get("workflow"):
    system_prompt_parts.append("")
    system_prompt_parts.append("## Workflow")
    for i, step in enumerate(instructions["workflow"], 1):
        system_prompt_parts.append(f"{i}. {step.get('instruction', step.get('action', str(step)))}")

if instructions.get("body"):
    system_prompt_parts.append("")
    system_prompt_parts.append(instructions["body"])

system_prompt = "\n".join(system_prompt_parts)

print("=== Real API Key Quality Test ===")
print(f"Agent: {identity['display_name']} v{identity['version']}")
print(f"Provider: {config['providers']['default']}")
print(f"Model: {config['providers']['deepseek']['default_model']}")

# ── Open LLM client ──
client = LLMClient(
    provider="deepseek",
    api_key=config["providers"]["deepseek"]["api_key"],
    base_url=config["providers"]["deepseek"]["base_url"],
    model=config["providers"]["deepseek"]["default_model"],
    timeout=60,
)

# ── Test queries ──
test_queries = [
    "What is the weather like in Tokyo today? Please describe the expected temperature, conditions, and any recommendations. Be specific with numbers (e.g., '25°C' not 'warm').",  # noqa: E501
    "Compare the weather in Beijing vs Shanghai for today. Which city would be better to visit?",
]

for idx, query in enumerate(test_queries, 1):
    print(f"\n{'=' * 60}")
    print(f"Test {idx}/{len(test_queries)}")
    print(f"{'=' * 60}")
    print(f"Query ({len(query)} chars): {query[:100]}...")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    response_text = client.chat(
        model=config["providers"]["deepseek"]["default_model"],
        messages=messages,
        max_tokens=500,
    )

    print(f"\nResponse ({len(response_text)} chars):")
    print("-" * 40)
    print(response_text)
    print("-" * 40)

    # ── Quality assessment ──
    checks = []

    # 1. Substantive length
    if len(response_text) > 30:
        checks.append(("PASS", f"Substantive ({len(response_text)} chars)"))
    else:
        checks.append(("FAIL", f"Too short ({len(response_text)} chars)"))

    # 2. Addresses topic
    topic_words = ["weather", "temperature", "Tokyo", "Beijing", "Shanghai", "degree", "condition"]
    found = [w for w in topic_words if w.lower() in response_text.lower()]
    if len(found) >= 2:
        checks.append(("PASS", f"Addresses topic (found: {found[:4]})"))
    else:
        checks.append(("WARN", f"Weak topic match (found: {found})"))

    # 3. Specific data
    numbers = re.findall(r'\d+', response_text)
    if numbers:
        checks.append(("PASS", f"Contains specific data ({len(numbers)} numbers)"))
    else:
        checks.append(("WARN", "No numerical data"))

    # 4. Multi-sentence structure
    sentences = [s.strip() for s in response_text.replace('!', '.').replace('?', '.').split('.') if len(s.strip()) > 5]  # noqa: E501
    if len(sentences) >= 2:
        checks.append(("PASS", f"Multi-sentence ({len(sentences)} sentences)"))
    else:
        checks.append(("WARN", "Too brief (single thought)"))

    # 5. No refusal
    refusal = ["I am unable", "I cannot", "I don't have", "sorry", "apologize"]
    has_refusal = any(p.lower() in response_text.lower() for p in refusal)
    if not has_refusal:
        checks.append(("PASS", "No refusal patterns"))
    else:
        checks.append(("WARN", "Contains hedging/refusal"))

    for status, msg in checks:
        print(f"  [{status}] {msg}")

    print(f"  Usage: {client.last_usage}")

print(f"\n{'=' * 60}")
print("v0.8.0 Gap Summary")
print(f"{'=' * 60}")
print("Current state:")
print("  Agent responds substantively using LLM knowledge of weather")
print("  Tools are stubs (weather_current/weather_forecast return placeholders)")
print("  Script query_weather.sh is subprocess-based, not import-bound")
print("")
print("v0.8.0 target:")
print("  Phase 1.5 extracts function signatures from scripts/*.py")
print("  Agent calls typed functions via import binding (not subprocess)")
print("  CapBus.register() with real JSON Schema from AST analysis")
print("  Real API data flows through typed function calls")

print("\nDone.")
