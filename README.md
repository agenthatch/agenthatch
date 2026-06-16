<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/agenthatch/.github/main/profile/assets/logo-dark.svg">
    <img alt="agenthatch" src="https://raw.githubusercontent.com/agenthatch/.github/main/profile/assets/logo-light.svg" width="600">
  </picture>
</p>

<p align="center">
  <strong>Turn any SKILL.md into a standalone, runnable AI Agent.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/agenthatch/">
    <img src="https://img.shields.io/pypi/v/agenthatch?color=blue" alt="PyPI version">
  </a>
  <a href="https://pypi.org/project/agenthatch/">
    <img src="https://img.shields.io/pypi/pyversions/agenthatch" alt="Python versions">
  </a>
  <a href="https://pypi.org/project/agenthatch/">
    <img src="https://img.shields.io/pypi/dm/agenthatch" alt="PyPI downloads">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  </a>
  <a href="https://discord.gg/TODO">
    <img src="https://img.shields.io/badge/Discord-coming_soon-5865F2" alt="Discord">
  </a>
  <a href="https://twitter.com/TODO">
    <img src="https://img.shields.io/badge/Twitter-coming_soon-1DA1F2" alt="Twitter">
  </a>
</p>

---

## The problem with SKILL.md

SKILL.md promised a lot. Write a markdown file, tell your agent what to do, and
it works. In practice, anyone who has used more than three skills across Claude
Code, Codex CLI, or OpenClaw knows the friction:

| Pain point | What actually happens |
|---|---|
| **No isolation** | Skills leak into each other. A file-organizer skill and a git-ops skill share the same context window. The agent confuses instructions meant for one with the other. |
| **Reference book, not operating manual** | Agents treat SKILL.md as a loose suggestion, not a contract. Given a long skill, the model skim-reads it. It picks the parts that seem relevant and ignores the rest. |
| **Token waste** | Every SKILL.md lives in the system prompt. Add 5 skills at 3KB each and you just burned 15KB of context before the conversation even starts. On long-running tasks this compounds fast. |
| **No validation** | A typo in a tool name, a missing parameter, an ambiguous instruction. The agent won't catch any of it until runtime, and by then the conversation is 20 turns deep. |
| **Scale decays** | Skills work at 1–3. At 10+ they become unmanageable. No dependency graph, no conflict detection, no way to tell which skill overrides which. |

The core issue isn't the format. It's that SKILL.md is **prompt engineering**, not
**software engineering**. You're asking an LLM to interpret human prose at
runtime, every time, with no compilation, no type checking, no contract.

---

## What agenthatch does

agenthatch treats a SKILL.md as **source code** — not a prompt. It compiles it
through a deterministic pipeline into a standalone Python agent that you can
import, ship, and run anywhere.

```
SKILL.md  →  Parse  →  6-Harness LLM Pipeline  →  Code Generation  →  Runnable Agent
   (input)   (Phase 1)    (Phase 2: AI inference)     (Phase 3: Jinja2)     (output)
```

The result is a self-contained Python package with its own `pyproject.toml`,
a CLI entry point, typed tool definitions, MCP integration, and a runtime
configuration. It's not a wrapper around your skill — it **is** the skill,
compiled into code.

---

## Demo

<!-- TODO: record terminal demo GIF showing the full pipeline:
     agenthatch init → skills add → hatch → run
     Recommended: asciinema or terminalizer for SVG terminal recording -->

<!-- TODO: record a longer demo showing a multi-turn agent conversation
     with PlanLayer-driven execution, tool calling, and MCP integration -->

<p align="center">
  <em>Demo coming soon. For now, try the Quick Start below — it takes under a minute.</em>
</p>

---

## Quick Start

```bash
# Install
pip install agenthatch

# Initialize with your LLM provider
agenthatch init

# Add a SKILL.md
agenthatch skills add ./my-skill/SKILL.md

# Hatch it into an agent
agenthatch hatch my-skill

# Run it
agenthatch run my-skill
```

Three steps from markdown to running agent. The hatched agent lives in your
skillhouse and can be re-run anytime.

---

## SKILL.md vs agenthatch

| | SKILL.md (raw) | agenthatch (hatched) |
|---|---|---|
| **Execution** | Interpreted at runtime by LLM | Compiled into standalone Python package |
| **Isolation** | All skills share one context window | Each agent has its own runtime, tools, and config |
| **Validation** | None. Typos and ambiguities caught at runtime. | Schema-validated AHSSPEC before code generation |
| **Token cost** | Full skill body in system prompt every turn | ~150 bytes of runtime config |
| **Tool definitions** | Prose descriptions, LLM guesses how to call | Type-annotated Python functions with JSON Schema |
| **MCP** | Manual wiring per agent | Auto-detected, auto-configured |
| **Determinism** | LLM interprets differently each time | Same SKILL.md → same agent binary |
| **Multi-skill scaling** | Degrades past 3–5 skills | Unlimited. Each agent is a separate process. |
| **Debugging** | Read the LLM's chain-of-thought and pray | Standard Python debugging, logging, tests |

---

## Architecture

<!-- TODO: architecture diagram showing the 3-phase pipeline
     (Parse → 6-Harness LLM → Code Generation) -->

agenthatch runs a **3-phase pipeline** with 6 AI harnesses working in parallel:

### Phase 1: Deterministic Parse (no AI)

The SKILL.md is parsed for frontmatter, body, and directory files. No AI
involved. A pure file-system operation. The output is a `ContextPack`
with zero semantic transformation.

### Phase 2: 6-Harness LLM Inference

Six specialized AI harnesses run concurrently, each with its own persona,
temperature, and model tier:

| Harness | Role | Model | Temp |
|---|---|---|---|
| **A — Identity** | Extract name, version, description from frontmatter | Small | 0.1 |
| **B — Intent** | Infer trigger phrases and user intents | Small | 0.5 |
| **C — Interface** | Design tool signatures, parameters, and return types | Large | 0.5 |
| **D — Base** | Detect runtime base class and instruction structure | Large | 0.3 |
| **E — Assembly** | Cross-validate all harness outputs, produce AHSSPEC | Small | 0.2 |
| **F — MCP** | Detect and configure MCP server connections | Moderate | 0.3 |

Each harness runs an **Analyze → Infer → Self-Validate → Correct** loop with
up to 2 internal retries. Harness E cross-validates the other five outputs and
produces a unified AHSSPEC (Agent Hatch Standard Specification).

### Phase 3: Code Generation

Jinja2 templates render the AHSSPEC into a complete Python agent package:

```
hatched-agent/
├── pyproject.toml          # pip-installable package
├── agent.py                # Agent class (extends AHCoreAgent)
├── cli.py                  # Typer-based CLI entry point
├── tools.py                # Type-annotated tool implementations
├── runtime.toml            # LLM provider, model, API keys
└── README.md               # Generated usage docs
```

### Runtime: PlanLayer

Generated agents use the **PlanLayer state machine** — a 6-state planning
engine that runs STARTING → PLANNING → EXECUTING → VERIFYING → REPLANNING →
DONE. It adapts mid-task: merges completed steps, branches on failure, and
degrades gracefully when tools time out.

---

## How it works under the hood

<details>
<summary>Click to expand: the full pipeline in detail</summary>

### Step 1: `agenthatch init`

Sets up `~/.agenthatch/` with your LLM provider configuration. Supports OpenAI,
DeepSeek, Anthropic, and any OpenAI-compatible endpoint. The config file is
TOML. Readable, versionable, easy to share.

### Step 2: `agenthatch skills add <path>`

Copies the SKILL.md and its directory into the skillhouse index. The skillhouse
tracks every skill you've added, its hatch status, and where its generated agent
lives.

### Step 3: `agenthatch hatch <name>`

The full pipeline runs:

```
Phase 1 (deterministic): Parse SKILL.md → ContextPack
Phase 2 (AI): 6 harnesses → HarnessOutput → Assembly → AHSSPEC
Phase 3 (Jinja2): AHSSPEC → agent package
```

Flags:
- `--no-generate` — skip Phase 3, review the AHSSPEC first
- `--force` — overwrite existing hatched agent
- `--dry-run` — preview without writing files

### Step 4: `agenthatch run <name>`

Launches the hatched agent in interactive TUI mode. The agent loads its
runtime config, connects to its LLM provider, and starts a conversation loop
with tool calling, context compaction, and PlanLayer-driven execution.

</details>

---

## CLI Reference

| Command | What it does |
|---|---|
| `agenthatch init` | Initialize config and provider setup |
| `agenthatch skills add <path>` | Register a SKILL.md in the skillhouse |
| `agenthatch skills list` | List all registered skills |
| `agenthatch skills delete <name>` | Remove a skill from the skillhouse |
| `agenthatch hatch <name>` | Run the full pipeline (parse → harness → generate) |
| `agenthatch run <name>` | Launch a hatched agent in interactive TUI |
| `agenthatch search <query>` | Search the skillhouse index |
| `agenthatch doctor` | Diagnose environment and dependencies |
| `agenthatch assemble` | Re-assemble an existing skillhouse agent |

---

## Installation

```bash
pip install agenthatch
```

Requires Python 3.11 or later.

For development:

```bash
git clone https://github.com/agenthatch/agenthatch.git
cd agenthatch
pip install -e ".[dev]"
```

---

## Documentation

<!-- TODO: set up docs site (MkDocs Material or Mintlify) -->

| Document | Link |
|---|---|
| Contributing Guide | [CONTRIBUTING.md](CONTRIBUTING.md) |
| Security Policy | [SECURITY.md](SECURITY.md) |
| Support | [SUPPORT.md](SUPPORT.md) |
| Roadmap | [ROADMAP.md](ROADMAP.md) |
| Code of Conduct | [CODE_OF_CONDUCT.md](https://github.com/agenthatch/.github/blob/main/CODE_OF_CONDUCT.md) |
| Changelog | [CHANGELOG.md](CHANGELOG.md) |

---

## Community

- [GitHub Discussions](https://github.com/agenthatch/agenthatch/discussions) — questions, ideas, roadmap
- [GitHub Issues](https://github.com/agenthatch/agenthatch/issues) — bugs and feature requests
- Discord — coming soon
- X/Twitter — coming soon

---

## Contributing

agenthatch is a solo project looking for its first contributors.
Issues, pull requests, documentation, design -- every bit moves the project forward.

See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, the quality gate
(`hatch run quality:check`), and PR guidelines.

AI-assisted contributions are welcome. Run the quality gate before submitting —
that's all that matters.

---

## FAQ

### Who is this for?

Anyone who maintains more than 3 SKILL.md files and feels the friction. Claude
Code users, Codex CLI users, OpenClaw users — if you've ever thought "I wish
this skill was a real program," this is for you.

### Can I use this with Claude Code / Codex / OpenClaw?

Yes. The hatched agent is a standalone Python package. You can run it as a CLI,
import it as a library, or wrap it as an MCP server. It doesn't depend on any
specific agent platform.

### What MCP servers are supported?

Any MCP server that speaks the standard protocol. Harness F auto-detects MCP
servers referenced in your SKILL.md and configures them in the generated agent's
runtime.

### Does this replace SKILL.md?

No. SKILL.md is the input format. agenthatch is the compiler. You still write
skills in markdown — agenthatch turns them into agents.

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

<sub>📖 简体中文版请见 [README_CN.md](README_CN.md)</sub>