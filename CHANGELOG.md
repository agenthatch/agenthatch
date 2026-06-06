# CHANGELOG

All notable changes to agenthatch will be documented in this file.

---

## [v0.6.0] — 2026-06-05

### Architecture Transformation: "Agent Factory"

v0.6 marks a major architectural transformation from "configuration-driven" to "Agent Factory" mode. The core runtime has been extracted into a standalone `agenthatch-core` package, and the `hatch` command now includes built-in Phase 3 agent generation.

### Added

- **agenthatch-core**: New standalone package providing the universal agent runtime base
  - `AHCoreAgent`: Base class for all generated agents
  - `LLMClient`: Unified LLM call interface (OpenAI, DeepSeek, custom providers)
  - `CapBus`: Capability bus for tool registration, routing, and execution
  - `Sandbox`: Subprocess sandbox with command whitelisting and timeout control
  - `ConversationLoop`: LLM ↔ Tool conversation loop with circuit breaker and retry
  - `ContextManager`: System prompt builder, history management, auto-compaction
  - `CompactSummary`: LLM-generated structured context compression
  - `resolve_runtime_config()`: Environment variable resolver with `${VAR}` syntax
- **Phase 3 Agent Generation**: `hatch` command now generates standalone, independently-runnable Agent directories
  - Jinja2 template engine with 6 templates (pyproject.toml, agent.py, cli.py, tools.py, runtime.toml, README.md)
  - `GenerateEngine` class for extracting AHSSPEC variables and rendering templates
  - `generate_agent()` convenience function
- **`agenthatch run` redesign**: Direct agent launching via in-process import with Rich Live TUI
  - Three-level agent discovery: current dir → skillhouse index → user dir
  - Interactive commands: `/help`, `/compact`, `/clear`, `/quit`
- **`agenthatch migrate`**: New command for migrating v0.5 agenthatch.yaml to v0.6 format
- **`agent.status` and `agent.generated_at` fields**: New metadata fields in agenthatch.yaml
- **`agent_output` field**: New field in skillhouse index for tracking agent generation paths

### Changed

- **agenthatch.yaml format**: Runtime configuration (`agent.runtime.*`) removed and migrated to `runtime.toml`
- **`hatch` command**: Now executes full 3-phase pipeline by default (parse → harness → generate)
  - `--no-generate` flag to skip Phase 3 (review mode)
  - `--force` flag to overwrite existing output
  - `--dry-run` flag to preview without writing
  - `--no-copy-skills` flag to exclude original SKILL.md
- **Dependency architecture**: `agenthatch` now depends on `agenthatch-core>=0.6.0` (one-way dependency)
- **`ConversationLoop`**: Migrated to `agenthatch-core`, now receives `llm`, `capbus`, `sandbox`, `ctx` as constructor parameters
- **`ContextManager`**: Migrated to `agenthatch-core`, accepts `dict` or `SpecProtocol` for spec
- **`LLMClient`**: Migrated to `agenthatch-core`, accepts provider details directly

### Fixed

- **Fix-1**: hatch exit code verification — exit code is now always 0 on success
- **Fix-2**: init command version number — now reads from `agenthatch.__version__` instead of hardcoded string
- **Fix-3**: skills list display — unhatched skills now show `[dim]not hatched[/dim]` instead of `Version ?`
- **Fix-4**: reasoning_content handling — verified DeepSeek V4 Pro streaming with reasoning content fallback
- **Fix-5**: TUI backspace key — Rich Live context properly paused during `Prompt.ask()` input
- **Fix-6**: legacy `run` command logic — removed configuration-driven path, replaced with agent direct-launch

### Removed

- **`agent.runtime` fields** from agenthatch.yaml (provider, model, api_key, temperature, max_tokens, features, compact)
- **Legacy `SkillAgent.from_ahspec()` runtime assembly path**: Replaced by `AHCoreAgent` + generated agent code
- **Configuration-driven `run` path**: Replaced by agent direct-launch mode

### Deprecated

- `agent.runtime` in agenthatch.yaml: Issues `DeprecationWarning` on load, still functional
- Will be removed in v1.0.0 per the deprecation schedule

---

## [v0.5.10] — 2026-05-XX

### Fixed
- DD-09-01: `from_openai()` empty response handling
- DD-09-02: CompactSummary checkpoint TypeError
- DD-09-03: max_tokens 0.7x → 2.5x inflation
- DD-09-04: MCP Server URL extraction
- DD-09-05: reasoning_content JSON extraction
- DD-09-06: chat_structured() reasoning fallback
- DD-09-07: Harness E structured confidence
- DD-09-08: Token adjustment log level
- DD-09-09: _extract_content() multi-format awareness

---

## [v0.5.0] — 2025-XX-XX

### Added
- SKILL.md parsing (Phase 1): Deterministic frontmatter + content parsing
- LLM Harness reasoning (Phase 2): 6-agent harness pipeline for AHSSPEC generation
- agenthatch.yaml output: Structured skill specification
- `agenthatch hatch` command: SKILL.md → agenthatch.yaml pipeline
- `agenthatch run` command: Interactive agent conversation
- `agenthatch init` command: First-time setup wizard
- `agenthatch skills` command: Skill listing and management
- Skillhouse index: Skill discovery and registration
- Semantic search: sentence-transformers based skill retrieval
- Rich TUI: Live streaming with tool call visualization

---

## [v0.2.0] — 2025-XX-XX

### Added
- Initial project scaffolding
- Basic CLI framework with Typer
- Configuration management