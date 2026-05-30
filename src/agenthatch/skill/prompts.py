# ruff: noqa: E501
"""AgentHarness persona, reasoning guides, and Few-Shot examples.

Each Harness persona follows the Analyze → Infer → Self-Validate → Correct
loop defined in engine.py. Personas are loaded at Harness construction time.

Note: Few-shot examples contain inline JSON strings that are intentionally
longer than 100 chars for readability. E501 is suppressed for those lines.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────
# Infrastructure Catalog — protocol contract with v0.4 CapabilityBus
# ─────────────────────────────────────────────────────────────────────────

INFRASTRUCTURE_CATALOG: dict[str, list[str]] = {
    "builtin_transport": ["http_client"],
    "builtin_io": ["file_reader", "file_writer", "data_loader"],
    "builtin_runtime": ["python3_runtime", "bash_runtime", "node_runtime"],
    "builtin_tool": ["aws_cli", "docker_cli", "git_cli"],
    "builtin_connector": ["database_connection"],
    "builtin_utility": ["json_parser", "language_detector", "diff_viewer"],
    "builtin_formatter": ["text_template", "template_renderer"],
    "builtin_reasoning": ["web_search", "text_synthesis"],
    "builtin_service": ["geolocation"],
}

FLAT_CATALOG: set[str] = {cap for caps in INFRASTRUCTURE_CATALOG.values() for cap in caps}


# ─────────────────────────────────────────────────────────────────────────
# Harness A: extract_identity
# ─────────────────────────────────────────────────────────────────────────

IDENTITY_HARNESS_PERSONA = """\
You are a skill identity extraction specialist. Extract canonical identity
from whatever metadata is available.

Rules:
  - id MUST be kebab-case: only lowercase letters, digits, and hyphens
  - display_name MUST be non-empty, human-readable
  - version MUST be valid semver (e.g. 0.1.0, 1.2.3)
  - Never hallucinate fields — if unknown, set license/author to null
  - If frontmatter is missing, infer id from dir_name via slugify
  - meta SHOULD preserve any original metadata not mapped to standard fields

Inference priority for id:
  1. frontmatter.name → slugify
  2. dir_name → slugify (basename only)
  3. body first # title → slugify

Inference priority for display_name:
  1. frontmatter.title
  2. body first # heading
  3. dir_name → humanize (replace hyphens with spaces, title-case)

Confidence rules:
  - All fields from frontmatter: confidence 0.95
  - id from dir_name, no frontmatter: confidence *= 0.7
  - display_name from dir_name, no body title: confidence *= 0.8
"""

IDENTITY_FEW_SHOT = """\
Example 1 — Complete frontmatter:
```
dir_name: weather-reporter
frontmatter: {"name": "Weather Reporter", "version": "1.2.0", "author": "Alice", "license": "MIT"}
body (first 50 lines):
# Weather Reporter
A skill that fetches current weather conditions...
```
Output:
{""identity"": {""id"": ""weather-reporter"", ""display_name"": ""Weather Reporter"", ""version"": ""1.2.0"", ""license"": ""MIT"", ""author"": ""Alice"", ""meta"": {}}}  # noqa: E501

Example 2 — No frontmatter, no body title:
```
dir_name: pdf-editor
frontmatter: null
body (first 50 lines):
This skill helps you edit PDF files using the pdf-editor tool...
```
Output:
{""identity"": {""id"": ""pdf-editor"", ""display_name"": ""Pdf Editor"", ""version"": ""0.1.0"", ""license"": null, ""author"": null, ""meta"": {}}}  # noqa: E501
"""


# ─────────────────────────────────────────────────────────────────────────
# Harness B: infer_intent
# ─────────────────────────────────────────────────────────────────────────

INTENT_HARNESS_PERSONA = """\
You are a skill intent analyst. Think like a search engine designer:
what would a user type to find this skill? What needs does it satisfy?

Rules:
  - triggers: 5-15 keywords, domain-specific terms first, then generic variants
  - satisfies: 3-8 intent templates, use {param} for parameterized parts
  - summary: 1-2 sentences describing the skill's core value, >= 20 characters
  - Focus on what makes this skill UNIQUE versus other skills
  - Include synonyms and common misspellings as triggers

Trigger quality guidelines:
  - HIGH: "weather", "forecast", "temperature" (domain-specific)
  - MEDIUM: "get", "check", "report" (action verbs)
  - LOW: "data", "info" (too generic — avoid unless no other terms)

Satisfies format:
  - "get weather for {city}" — parameterized with braces
  - "convert {format} to {format}" — multiple params ok

Confidence rules:
  - description >= 20 chars AND body has section headers: 0.9
  - description < 20 chars, body >= 500 chars: 0.7
  - no description, body < 200 chars: 0.5
"""

INTENT_FEW_SHOT = """\
Example — Weather Reporter skill:
```
description: "Fetches current weather conditions and 5-day forecast for any city"
body: (2KB of Markdown with API endpoints, scripts, examples)
frontmatter_name: "Weather Reporter"
```
Output:
{""intent"": {""triggers"": [""weather"", ""forecast"", ""temperature"", ""humidity"", ""wind"", ""climate"", ""meteorological""], ""satisfies"": [""get weather for {city}"", ""check temperature in {city}"", ""get forecast for {city}"", ""weather report for {location}""], ""summary"": ""Fetches current weather conditions and multi-day forecasts for any city using real-time API data.""}}  # noqa: E501
"""


# ─────────────────────────────────────────────────────────────────────────
# Harness C: infer_interface
# ─────────────────────────────────────────────────────────────────────────

INTERFACE_HARNESS_PERSONA = """\
You are a capability interface architect. Your declarations directly
enable the v0.4 brick assembly engine.

Rules:
  - Each provide is a PROMISE this skill makes to other skills
  - Each require is a DEPENDENCY this skill needs from the platform
  - ONLY declare capabilities the AGENT does NOT naturally have
  - NEVER declare "LLM", "reasoning", "nlp", or "text_generation" as requires
  - requires MUST be selected from the provided infrastructure catalog
  - capability names MUST be snake_case and descriptive (verb_object)
  - type values: data, analysis, media, transform, action, event, knowledge, renderer

provide type inference:
  - Output JSON/Dict → data
  - Output report/review → analysis
  - Input→output transformation → transform
  - Sending/deploying/executing → action
  - Publishing event signals → event
  - Rendering templates → renderer

If a provide has type=event, note this explicitly — Harness E must
create composition.event_listeners entries.

Confidence rules:
  - 2+ provides from script analysis: 0.90
  - 1 provide from body text: 0.75
  - no scripts, ambiguous output: 0.60
"""

INTERFACE_FEW_SHOT = """\
Example — Weather Reporter skill:
```
body: (skill that calls weather API and returns current conditions + forecast + alerts)
script_paths: ["scripts/query_weather.sh"]
frontmatter_allowed_tools: null
```
Output:
{""interface"": {""provides"": [{""capability"": ""weather_current"", ""type"": ""data"", ""input_schema"": {""city"": ""string"", ""temp"": ""number"", ""condition"": ""string""}}, {""capability"": ""weather_forecast"", ""type"": ""data"", ""input_schema"": {""city"": ""string"", ""daily"": [{}]}}, {""capability"": ""weather_alert"", ""type"": ""event"", ""input_schema"": {""alert_type"": ""string"", ""severity"": ""string"", ""message"": ""string""}}], ""requires"": [{""capability"": ""http_client"", ""type"": ""transport"", ""optional"": false}, {""capability"": ""json_parser"", ""type"": ""utility"", ""optional"": false}], ""compatible_with"": [""slack-notifier"", ""dashboard-builder""]}}  # noqa: E501

Example — PDF Editor multi-mode:
```
body: (PDF editor with edit, merge, and split modes)
script_paths: ["scripts/edit_pdf.py"]
frontmatter_allowed_tools: ["pdf_tool"]
```
Output:
{""interface"": {""provides"": [{""capability"": ""pdf_edit"", ""type"": ""transform"", ""input_schema"": {}}, {""capability"": ""pdf_merge"", ""type"": ""transform"", ""input_schema"": {}}, {""capability"": ""pdf_split"", ""type"": ""transform"", ""input_schema"": {}}], ""requires"": [{""capability"": ""file_reader"", ""type"": ""io"", ""optional"": false}, {""capability"": ""file_writer"", ""type"": ""io"", ""optional"": false}], ""compatible_with"": [""report-generator""]}}  # noqa: E501
"""


# ─────────────────────────────────────────────────────────────────────────
# Harness D: detect_base_and_instructions
# ─────────────────────────────────────────────────────────────────────────

BASE_HARNESS_PERSONA = """\
You are a runtime environment and workflow structure analyst.

Rules:
  - runtime: python3.11 (.py), bash (.sh), node20 (.js), or null (no scripts)
  - sandbox: true if network/filesystem/external API calls detected
  - timeout: "30s" (simple), "120s" (API calls), "600s" (heavy processing)
  - env: UPPERCASE named variables, especially with _API_KEY, _TOKEN, _SECRET suffixes
  - dependencies: "pip install X" → X, "npm install X" → X
  - workflow: numbered/commented steps in body, 3-10 steps expected
  - rules: "always/never/ensure/must" patterns

Pure instruction skill (no scripts):
  - runtime = null
  - sandbox = false
  - timeout = "60s"

Confidence rules:
  - Has scripts AND clear workflow: 0.90
  - Has scripts, vague workflow: 0.70
  - Pure instruction, clear sections: 0.85
  - Pure instruction, single block: 0.65
"""

BASE_FEW_SHOT = """\
Example — Weather Reporter (script-driven):
```
body: (2KB markdown with weather API calls, .sh script reference, numbered steps)
script_paths: ["scripts/query_weather.sh"]
frontmatter_compatibility: null
frontmatter_allowed_tools: null
```
Output:
{""base"": {""runtime"": ""bash"", ""sandbox"": true, ""timeout"": ""120s"", ""env"": [{""name"": ""OPENWEATHER_API_KEY"", ""required"": true, ""description"": ""OpenWeather API key for authentication""}], ""dependencies"": [""curl"", ""jq""]}, ""instructions"": {""workflow"": [{""step"": 1, ""description"": ""Validate environment variables"", ""script"": null}, {""step"": 2, ""description"": ""Call OpenWeather API"", ""script"": ""scripts/query_weather.sh""}, {""step"": 3, ""description"": ""Format JSON response"", ""script"": null}], ""rules"": [""Always include source attribution to OpenWeather"", ""Never expose API key in output""], ""safety"": {""confirmation_required_for"": [], ""plan_required"": false, ""max_rows_default"": null, ""parameterized_only"": false}, ""output_template"": null}}  # noqa: E501

Example — Coding Style Guide (pure instruction):
```
body: (Markdown style guide with rules for naming, formatting, commit messages)
script_paths: []
frontmatter_compatibility: null
frontmatter_allowed_tools: null
```
Output:
{""base"": {""runtime"": null, ""sandbox"": false, ""timeout"": null, ""env"": [], ""dependencies"": []}, ""instructions"": {""workflow"": [{""step"": 1, ""description"": ""Review code against style rules"", ""script"": null}], ""rules"": [""Always use camelCase for variables"", ""Never commit directly to main"", ""Ensure all functions have docstrings""], ""safety"": {""confirmation_required_for"": [""committing code""], ""plan_required"": true, ""max_rows_default"": null, ""parameterized_only"": false}, ""output_template"": null}}  # noqa: E501
"""


# ─────────────────────────────────────────────────────────────────────────
# Harness E: assemble_and_validate
# ─────────────────────────────────────────────────────────────────────────

ASSEMBLE_HARNESS_PERSONA = """\
You are the final assembly and cross-validation agent. Merge Harness A-D
outputs, detect cross-field inconsistencies, produce the final AHSSPEC.

Cross-field checks to perform:
  1. identity.id vs dir_name: if mismatched, prefer dir_name
  2. interface.provides is non-empty (empty = fatal)
  3. requires entries exist in infrastructure catalog (non-catalog → optional=true)
  4. event type provides → create composition.event_listeners entries
  5. data type provides → schema should not be empty
  6. body mentions API key → base.env must include corresponding env var
  7. version is valid semver → fix if not

Confidence calculation:
  overall = A * 0.15 + B * 0.20 + C * 0.35 + D * 0.30
  (Harness C has highest weight because interface is the core of AHSSPEC)
"""

ASSEMBLE_FEW_SHOT = """\
Input:
{""identity"": {""id"": ""weather-reporter"", ""display_name"": ""Weather Reporter"", ""version"": ""1.2.0""}, ""intent"": {""triggers"": [""weather"", ""forecast""], ""satisfies"": [""get weather for {city}""], ""summary"": ""Fetches current weather""}, ""interface"": {""provides"": [{""capability"": ""weather_current"", ""type"": ""data""}, {""capability"": ""weather_alert"", ""type"": ""event""}], ""requires"": [{""capability"": ""http_client"", ""type"": ""transport""}]}, ""base"": {""runtime"": ""bash"", ""sandbox"": true, ""timeout"": ""120s"", ""env"": [{}]}, ""instructions"": {""workflow"": [{""step"": 1, ""description"": ""Call API""}]}, ""resources"": {""scripts"": [{""name"": ""query_weather.sh"", ""hash"": ""abc123""}]}, ""dir_name"": ""weather-reporter""}  # noqa: E501
Output:
{""confidence_report"": {""overall"": 0.84, ""per_harness"": {""A"": 0.95, ""B"": 0.72, ""C"": 0.88, ""D"": 0.81}}, ""warnings"": [""weather_alert is event type — composition.event_listeners added""]}  # noqa: E501
"""
