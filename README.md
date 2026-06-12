# agenthatch

**Turn any SKILL.md into a runnable AI Agent.**

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-alpha-orange)]()

agenthatch is an **Agent Factory** that transforms declarative `SKILL.md` files into fully functional, standalone AI agents. Inspired by the [Claude Code `SKILL.md` specification](https://docs.anthropic.com/en/docs/claude-code/skills), agenthatch goes further: it analyzes, reasons about, and generates production-ready agents with tool calling, MCP integration, and multi-turn conversation capabilities.

---

## Why agenthatch?

|                 | Claude Code + SKILL.md | agenthatch |
|-----------------|------------------------|------------|
| Agent format    | Inline prompt injection | Standalone runnable agent |
| Tool calling    | Built-in tools only     | MCP + custom tools + sandbox |
| Multi-turn      | Single-shot context     | Full conversation loop |
| Deployment      | Requires Claude Code    | Self-contained Python package |
| Customization   | None                   | Full harness pipeline |
| Quality control | Manual                 | Automated fidelity checks |

---

## Quick Start

### 1. Install

```bash
pip install agenthatch
```

### 2. Initialize

```bash
agenthatch init
```

### 3. Add a Skill

```bash
agenthatch skill add path/to/SKILL.md
```

### 4. Hatch an Agent

```bash
agenthatch hatch my-skill
```

This runs the full pipeline:
- **Phase 1** — Parse SKILL.md frontmatter and content
- **Phase 2** — 6-harness LLM reasoning pipeline (identity, intent, interface, base, assembly, MCP servers)
- **Phase 3** — Generate standalone agent code
- **Phase 4** — Readiness verification

### 5. Run the Agent

```bash
agenthatch run my-skill
```

---

## How It Works

### The Harness Pipeline

agenthatch uses a chain of specialized LLM agents ("harnesses") to analyze and reason about your skill:

```
SKILL.md
  │
  ├─ Harness A: Identity     → Who is this agent?
  ├─ Harness B: Intent       → What triggers and satisfies it?
  ├─ Harness C: Interface    → What capabilities does it provide/require?
  ├─ Harness F: MCP Servers  → What MCP connections does it need?
  ├─ Harness D: Base         → What runtime environment?
  ├─ Harness E: Assembly     → Cross-validate and produce AHSSPEC
  │
  ▼
agenthatch.yaml (AHSSPEC)
  │
  ▼
Generated Agent (standalone Python package)
```

### Fidelity Protection

Every generated agent includes:
- **Fidelity Anchors** — SHA256 hashes of constraints extracted from the original SKILL.md
- **Fidelity Manifest** — Verification file in the agent directory
- **Quality Review** — Harness E validates intent fidelity, capability coverage, and MCP integrity

### Skill Management

```bash
# List all skills
agenthatch skill list

# Add a new skill
agenthatch skill add path/to/SKILL.md

# Delete a skill
agenthatch skill delete my-skill

# Search skills
agenthatch search "data analysis"
```

---

## SKILL.md Format

agenthatch follows the [Claude Code SKILL.md specification](https://docs.anthropic.com/en/docs/claude-code/skills):

```markdown
---
name: My Skill
description: What this skill does
---

# Skill Instructions

Detailed instructions for the agent...

## Workflow

1. Step one
2. Step two

## MCP Tools

This skill uses mcp__my-server__my-tool for data access.
```

### MCP Support

agenthatch automatically detects MCP server references in your SKILL.md:

- `mcp__SERVER__TOOL` patterns
- `mcporter call Server.Tool` syntax
- Frontmatter `mcpServers` declarations

---

## Architecture

```
agenthatch/
├── src/agenthatch/          # CLI, skill engine, harness, generation
│   ├── cli/                 # Typer CLI commands
│   ├── skill/               # Skill parsing, harness, validation
│   ├── generate/            # Agent code generation + templates
│   ├── agent/               # Runtime, builtins, MCP
│   ├── house/               # Skillhouse index, discovery
│   └── config/              # Configuration management
├── agenthatch-core/         # Universal agent runtime
│   └── src/agenthatch_core/ # LLM client, sandbox, conversation loop
└── tests/                   # Test suite
```

---

## Requirements

- Python 3.11+
- LLM API access (OpenAI, DeepSeek, or custom provider)
- Optional: `mcporter` for MCP server support (`npm install -g mcporter`)

---

## Contributing

agenthatch is in active development. Contributions are welcome!

```bash
# Development setup
git clone https://github.com/agenthatch/agenthatch
cd agenthatch
pip install -e ".[dev]"

# Run tests
pytest

# Quality checks
hatch run quality:check
```

---

## License

MIT — see [LICENSE](LICENSE) for details.