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
    "builtin_io": ["file_reader", "file_writer"],
    "builtin_runtime": ["python3_runtime", "bash_runtime"],
    "builtin_utility": ["json_parser"],
    "builtin_formatter": ["template_renderer"],
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
  - version: omit this field (v0.8.9: version is deprecated, hatched_at is the canonical timestamp)
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
{""identity"": {""id"": ""weather-reporter"", ""display_name"": ""Weather Reporter"", ""license"": ""MIT"", ""author"": ""Alice"", ""meta"": {}}}  # noqa: E501

Example 2 — No frontmatter, no body title:
```
dir_name: pdf-editor
frontmatter: null
body (first 50 lines):
This skill helps you edit PDF files using the pdf-editor tool...
```
Output:
{""identity"": {""id"": ""pdf-editor"", ""display_name"": ""Pdf Editor"", ""license"": null, ""author"": null, ""meta"": {}}}  # noqa: E501
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
Example 1 — Weather Reporter (API-driven):
```
description: "Fetches current weather conditions and 5-day forecast for any city"
body: (2KB of Markdown with API endpoints, scripts, examples)
frontmatter_name: "Weather Reporter"
```
Output:
{""intent"": {""triggers"": [""weather"", ""forecast"", ""temperature"", ""humidity"", ""wind"", ""climate"", ""meteorological""], ""satisfies"": [""get weather for {city}"", ""check temperature in {city}"", ""get forecast for {city}"", ""weather report for {location}""], ""summary"": ""Fetches current weather conditions and multi-day forecasts for any city using real-time API data.""}}  # noqa: E501

Example 2 — Knowledge-Type (coding guide):
```
description: "Python coding style guide and best practices reference"
body: (3KB Markdown with naming conventions, formatting rules, commit message templates)
frontmatter_name: "Python Style Guide"
```
Output:
{""intent"": {""triggers"": [""style"", ""formatting"", ""convention"", ""naming"", ""pep8"", ""lint"", ""best practice"", ""code review""], ""satisfies"": [""check code style for {file}"", ""review {code} against conventions"", ""suggest naming for {function}"", ""format code according to style guide"", ""explain best practice for {topic}""], ""summary"": ""Provides Python coding style conventions, formatting rules, and best practices for maintaining consistent code quality across projects.""}}  # noqa: E501

Example 3 — Integration-Type (MCP connector):
```
description: "Connects to Notion knowledge base via MCP server for document search and management"
body: (1.5KB Markdown with mcp__notion__search, mcp__notion__read_document patterns)
frontmatter_name: "Notion Knowledge Base"
```
Output:
{""intent"": {""triggers"": [""notion"", ""knowledge base"", ""search"", ""document"", ""doc"", ""wiki"", ""find"", ""lookup"", ""query""], ""satisfies"": [""search knowledge base for {query}"", ""find documents about {topic}"", ""read document {id}"", ""create knowledge base entry"", ""list team spaces""], ""summary"": ""Connects to Notion knowledge management platform for searching, reading, creating, and managing documents across team spaces and knowledge bases.""}}  # noqa: E501

Example 4 — Pure Instruction (no scripts):
```
description: "Generate weekly project status reports from git activity"
body: (1KB Markdown with instructions for gathering commits, formatting, and sending)
frontmatter_name: "Weekly Status Reporter"
```
Output:
{""intent"": {""triggers"": [""status"", ""report"", ""weekly"", ""summary"", ""update"", ""progress"", ""recap"", ""standup""], ""satisfies"": [""generate weekly status report"", ""summarize project progress"", ""create sprint recap"", ""send status update to {team}""], ""summary"": ""Generates structured weekly project status reports by aggregating git activity, commit messages, and recent changes into a formatted summary.""}}  # noqa: E501
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
  - input_schema: use FLAT key→type format, NOT nested JSON Schema:
    ✓ {"doc_id": "string", "limit": "number"}
    ✗ {"type": "object", "properties": {}, "required": []}
    Valid types: "string", "number", "boolean", "array", "object"
    Leave empty {} only if the capability truly takes no parameters
  - output_schema: describe the return shape, at minimum include a "type" key
    ✓ {"type": "object", "properties": {"data": {"type": "string"}}}
    ✓ {"type": "string"}

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
  - MCP connector with tool names in body: 0.85
"""

INTERFACE_FEW_SHOT = """\
Example 1 — Weather Reporter (API-driven, shell script):
```
body: (skill that calls weather API and returns current conditions + forecast + alerts)
script_paths: ["scripts/query_weather.sh"]
frontmatter_allowed_tools: null
```
Output:
{""interface"": {""provides"": [{""capability"": ""weather_current"", ""type"": ""data"", ""input_schema"": {""city"": ""string"", ""temp"": ""number"", ""condition"": ""string""}}, {""capability"": ""weather_forecast"", ""type"": ""data"", ""input_schema"": {""city"": ""string"", ""daily"": [{}]}}, {""capability"": ""weather_alert"", ""type"": ""event"", ""input_schema"": {""alert_type"": ""string"", ""severity"": ""string"", ""message"": ""string""}}], ""requires"": [{""capability"": ""http_client"", ""type"": ""action"", ""optional"": false}, {""capability"": ""json_parser"", ""type"": ""transform"", ""optional"": false}], ""compatible_with"": [""slack-notifier"", ""dashboard-builder""]}}  # noqa: E501

Example 2 — PDF Editor (multi-mode, Python script):
```
body: (PDF editor with edit, merge, and split modes)
script_paths: ["scripts/edit_pdf.py"]
frontmatter_allowed_tools: ["pdf_tool"]
```
Output:
{""interface"": {""provides"": [{""capability"": ""pdf_edit"", ""type"": ""transform"", ""input_schema"": {""file_path"": ""string"", ""modifications"": ""object""}, ""output_schema"": {""type"": ""object"", ""properties"": {""output_path"": ""string"", ""pages_changed"": ""number""}}}, {""capability"": ""pdf_merge"", ""type"": ""transform"", ""input_schema"": {""files"": ""array"", ""output_name"": ""string""}, ""output_schema"": {""type"": ""object"", ""properties"": {""output_path"": ""string"", ""total_pages"": ""number""}}}, {""capability"": ""pdf_split"", ""type"": ""transform"", ""input_schema"": {""file_path"": ""string"", ""pages_per_chunk"": ""number""}, ""output_schema"": {""type"": ""array"", ""items"": {""type"": ""string""}}}], ""requires"": [{""capability"": ""file_reader"", ""type"": ""data"", ""optional"": false}, {""capability"": ""file_writer"", ""type"": ""data"", ""optional"": false}], ""compatible_with"": [""report-generator""]}}  # noqa: E501

Example 3 — MCP Connector (integration-type, multiple tools):
```
body: (Notion knowledge base connector with mcp__notion__search, mcp__notion__read_document, mcp__notion__create_document, mcp__notion__download_file)
script_paths: []
frontmatter_allowed_tools: null
```
Output:
{""interface"": {""provides"": [{""capability"": ""search_content"", ""type"": ""data"", ""input_schema"": {""query"": ""string"", ""limit"": ""number""}}, {""capability"": ""read_document"", ""type"": ""data"", ""input_schema"": {""doc_id"": ""string""}}, {""capability"": ""create_document"", ""type"": ""action"", ""input_schema"": {""title"": ""string"", ""content"": ""string""}}, {""capability"": ""download_file"", ""type"": ""data"", ""input_schema"": {""file_id"": ""string""}}], ""requires"": [{""capability"": ""http_client"", ""type"": ""action"", ""optional"": false}], ""compatible_with"": [""slack-notifier"", ""report-generator""]}}  # noqa: E501

Example 4 — Single-Script (no-script type, pure instruction):
```
body: (Weekly status report generator using git log analysis and markdown formatting)
script_paths: []
frontmatter_allowed_tools: null
```
Output:
{""interface"": {""provides"": [{""capability"": ""generate_report"", ""type"": ""transform"", ""input_schema"": {""period"": ""string"", ""repo"": ""string""}}, {""capability"": ""format_markdown"", ""type"": ""renderer"", ""input_schema"": {""content"": ""string""}}], ""requires"": [{""capability"": ""http_client"", ""type"": ""action"", ""optional"": true}], ""compatible_with"": [""slack-notifier"", ""email-sender""]}}  # noqa: E501
"""


# ─────────────────────────────────────────────────────────────────────────
# Harness D: detect_base_and_instructions
# ─────────────────────────────────────────────────────────────────────────

BASE_HARNESS_PERSONA = """\
You are a runtime environment and workflow structure analyst.

Rules:
  - runtime: python3.11 (.py), bash (.sh), node20 (.js), or null (no scripts)
  - timeout: "30s" (simple), "120s" (API calls), "600s" (heavy processing)
  - env: UPPERCASE named variables, especially with _API_KEY, _TOKEN, _SECRET suffixes
  - dependencies: "pip install X" → X, "npm install X" → X
  - workflow: numbered/commented steps in body, 3-10 steps expected
  - rules: "always/never/ensure/must" patterns

Pure instruction skill (no scripts):
  - runtime = null
  - timeout = "60s"

Confidence rules:
  - Has scripts AND clear workflow: 0.90
  - Has scripts, vague workflow: 0.70
  - Pure instruction, clear sections: 0.85
  - Pure instruction, single block: 0.65
"""

BASE_FEW_SHOT = """\
Example 1 — Weather Reporter (script-driven):
```
body: (2KB markdown with weather API calls, .sh script reference, numbered steps)
script_paths: ["scripts/query_weather.sh"]
frontmatter_compatibility: null
frontmatter_allowed_tools: null
```
Output:
{""base"": {""runtime"": ""bash"", ""timeout"": ""120s"", ""env"": [{""name"": ""OPENWEATHER_API_KEY"", ""required"": true, ""description"": ""OpenWeather API key for authentication""}], ""dependencies"": [""curl"", ""jq""]}, ""instructions"": {""workflow"": [{""step"": 1, ""description"": ""Validate environment variables"", ""script"": null}, {""step"": 2, ""description"": ""Call OpenWeather API"", ""script"": ""scripts/query_weather.sh""}, {""step"": 3, ""description"": ""Format JSON response"", ""script"": null}], ""rules"": [""Always include source attribution to OpenWeather"", ""Never expose API key in output""], ""safety"": {""confirmation_required_for"": [], ""plan_required"": false, ""max_rows_default"": null, ""parameterized_only"": false}, ""output_template"": null}}  # noqa: E501

Example 2 — Coding Style Guide (pure instruction):
```
body: (Markdown style guide with rules for naming, formatting, commit messages)
script_paths: []
frontmatter_compatibility: null
frontmatter_allowed_tools: null
```
Output:
{""base"": {""runtime"": null, ""timeout"": null, ""env"": [], ""dependencies"": []}, ""instructions"": {""workflow"": [{""step"": 1, ""description"": ""Review code against style rules"", ""script"": null}], ""rules"": [""Always use camelCase for variables"", ""Never commit directly to main"", ""Ensure all functions have docstrings""], ""safety"": {""confirmation_required_for"": [""committing code""], ""plan_required"": true, ""max_rows_default"": null, ""parameterized_only"": false}, ""output_template"": null}}  # noqa: E501

Example 3 — Mixed Script + Instruction (MCP connector with documents):
```
body: (1.5KB markdown with mcp__notion__search and mcp__notion__read_document patterns, numbered workflow steps)
script_paths: []
frontmatter_compatibility: null
frontmatter_allowed_tools: null
```
Output:
{""base"": {""runtime"": ""python3.11"", ""timeout"": ""120s"", ""env"": [{""name"": ""NOTION_TOKEN"", ""required"": true, ""description"": ""Notion API bearer token for authentication""}], ""dependencies"": [""requests""]}, ""instructions"": {""workflow"": [{""step"": 1, ""description"": ""List available knowledge bases"", ""script"": null}, {""step"": 2, ""description"": ""Search for relevant documents"", ""script"": null}, {""step"": 3, ""description"": ""Read selected document"", ""script"": null}, {""step"": 4, ""description"": ""Format and present results"", ""script"": null}], ""rules"": [""Always verify document exists before reading"", ""Never expose bearer token in output"", ""Ensure results are formatted as markdown tables""], ""safety"": {""confirmation_required_for"": [""create_document"", ""delete_document""], ""plan_required"": false, ""max_rows_default"": 50, ""parameterized_only"": false}, ""output_template"": null}}  # noqa: E501
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
  8. WORKFLOW-SCRIPT CONSISTENCY: For each workflow step that references a script
     by name, verify that script exists in resources.scripts. If a workflow step
     references a script_name, the corresponding script must be present in the
     discovered scripts list. Flag any workflow→script mismatches as warnings.

v0.8.2 — QUALITY REVIEW (perform AFTER assembly, before output):
  9. INTENT FIDELITY: Compare intent.summary against the original SKILL.md body
     included in the input. Does the summary accurately describe what the skill
     actually does? If the body describes a "document search" skill but the
     summary says "weather reporter", fix the summary.
  10. CAPABILITY COVERAGE: For every major workflow described in the skill body,
      there must be at least one provides entry and one workflow step. If a
      workflow is described in the body but missing from the assembled output,
      add it.
  11. MCP INTEGRITY: If interface.mcp_servers includes servers with non-empty
      command or transport fields, verify the server names match what is
      described in the body. If the body uses "mcporter call Notion.X" but
      mcp_servers has name "Notion" with empty command, it's wrong — fix it.
  12. TOOL FIDELITY: interface.provides should list capabilities that are
      actually described in the skill body, not fabricated ones. If a tool
      name appears in provides but is never mentioned in the body, flag it.

Confidence calculation:
  overall = A * 0.12 + B * 0.18 + C * 0.30 + D * 0.25 + F * 0.15
  (Harness C has highest weight because interface is the core of AHSSPEC;
   Harness F is included for MCP server detection accuracy)
"""

ASSEMBLE_FEW_SHOT = """\
Input:
{""identity"": {""id"": ""weather-reporter"", ""display_name"": ""Weather Reporter"", ""version"": ""1.2.0""}, ""intent"": {""triggers"": [""weather"", ""forecast""], ""satisfies"": [""get weather for {city}""], ""summary"": ""Fetches current weather""}, ""interface"": {""provides"": [{""capability"": ""weather_current"", ""type"": ""data""}, {""capability"": ""weather_alert"", ""type"": ""event""}], ""requires"": [{""capability"": ""http_client"", ""type"": ""action""}]}, ""base"": {""runtime"": ""bash"", ""timeout"": ""120s"", ""env"": [{}]}, ""instructions"": {""workflow"": [{""step"": 1, ""description"": ""Call API""}]}, ""resources"": {""scripts"": [{""name"": ""query_weather.sh"", ""hash"": ""abc123""}]}, ""dir_name"": ""weather-reporter""}  # noqa: E501
Output:
{""confidence_report"": {""overall"": 0.92, ""per_harness"": {""A"": 0.95, ""B"": 0.97, ""C"": 0.88, ""D"": 0.91, ""F"": 0.90, ""E"": 0.92}}, ""warnings"": [""weather_alert is event type — composition.event_listeners added""]}  # noqa: E501
"""


# ─────────────────────────────────────────────────────────────────────────
# Harness F: infer_mcp_servers
# ─────────────────────────────────────────────────────────────────────────

INFER_MCP_SERVERS_PROMPT = """You are analyzing a skill's documentation to identify MCP servers it depends on.

The user message contains the full SKILL.md body. Analyze it to find ALL MCP server references. Look for:
1. Pattern: `mcporter call SERVER_NAME.TOOL_NAME` — identifies mcporter-based MCP servers (command="mcporter", transport="stdio")
2. Pattern: `mcp__SERVER_NAME__TOOL_NAME` — extract SERVER_NAME
3. Pattern: `mcp_servers:` YAML sections in configuration blocks
4. Pattern: "MCP server" or "MCP service" mentions with connection details
5. Pattern: `requires: { "bins": ["mcporter"], "mcpServers": ["SERVER_NAME"] }` in frontmatter

For each MCP server found in the skill text, extract:
- name: server identifier (e.g., "data-infra-mcp", "ConfigHub-mcp", "Notion")
- transport: "stdio" if `mcporter call` is used, "streamable_http" if a URL/endpoint is given, "sse" if SSE
- url: full URL with http:// or https:// prefix (for HTTP/SSE transport), empty string for mcporter
- command: "mcporter" if the skill uses `mcporter call SERVER.TOOL` syntax, otherwise empty string
- description: what the server provides

CRITICAL RULES:
- If the skill uses `mcporter call X.Y` syntax, ALWAYS set command="mcporter" and transport="stdio"
- If the skill frontmatter has `"mcpServers": ["X"]`, detect server "X" with mcporter defaults
- Never fabricate a URL if no URL is given — set transport="" and url="" if unknown

Return ONLY valid JSON:
{"mcp_servers": [{"name": "...", "transport": "...", "url": "...", "command": "...", "description": "..."}]}

If no MCP servers found, return: {"mcp_servers": []}

## Few-Shot Examples

Example 1 — Skill with MCP server (HTTP transport):
```
This skill queries the data infrastructure via mcp__data-infra-mcp__query_database.
MCP server URL: http://localhost:8080/mcp
```
Output:
{"mcp_servers": [{"name": "data-infra-mcp", "transport": "streamable_http", "url": "http://localhost:8080/mcp", "command": "", "description": "Data infrastructure database query service"}]}

Example 2 — Skill with mcporter-based MCP server (stdio transport):
```
Uses ConfigHub MCP for configuration management. Run via: mcporter call ConfigHub-mcp.get_config key=app.settings
```
Output:
{"mcp_servers": [{"name": "ConfigHub-mcp", "transport": "stdio", "url": "", "command": "mcporter", "description": "Configuration management service"}]}

Example 3 — Skill with mcporter as forwarding layer, multiple tool calls:
```
This skill forwards requests through mcporter to the Notion MCP server.
mcporter call Notion.read_document resourceId="xxx"
mcporter call Notion.listSpaces type=1
```
Output:
{"mcp_servers": [{"name": "Notion", "transport": "stdio", "url": "", "command": "mcporter", "description": "Notion collaborative documentation platform MCP server"}]}

Example 4 — Skill with no MCP references:
```
A simple skill that formats JSON output from API responses.
```
Output:
{"mcp_servers": []}
"""
