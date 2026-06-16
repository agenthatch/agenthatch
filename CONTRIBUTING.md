# Contributing

Thanks for taking the time to contribute. This file tells you what you need
to start hacking on agenthatch.

Before anything else, read the [Code of Conduct](https://github.com/agenthatch/.github/blob/main/CODE_OF_CONDUCT.md).

---

## Getting started

### Prerequisites

- Python 3.11 or later
- [Git](https://git-scm.com/)

Optional but useful:

- **[mcporter](https://www.npmjs.com/package/mcporter)** — for MCP server testing
  during development. Install with `npm install -g mcporter`.

### Setup

```bash
git clone https://github.com/agenthatch/agenthatch.git
cd agenthatch
pip install -e ".[dev]"
```

The dev install pulls in pytest, ruff, mypy, build, and twine — the full quality
toolchain. You won't need these as an end user, but you do if you're changing
code or running tests.

Verify everything works:

```bash
agenthatch --help
```

If you see the CLI help text, you're good.

### Project structure

agenthatch is a Python monorepo with two packages under one build:

```
agenthatch/
├── src/agenthatch/                  # CLI, skill engine, harness, code generation
│   ├── cli/                         # Typer commands (init, hatch, run, etc.)
│   │   └── commands/
│   ├── skill/                       # SKILL.md parsing, 6-harness LLM pipeline
│   ├── generate/                    # Jinja2 templates, Agent code generation
│   ├── agent/                       # Runtime, built-in tools, MCP
│   ├── house/                       # Skillhouse index and discovery
│   ├── config/                      # Config loading and validation
│   ├── cap/                         # Capability bus and marshal
│   ├── base/                        # Sandbox base classes
│   ├── output/                      # Output sanitization
│   ├── providers.py                 # LLM provider abstraction
│   ├── exceptions.py                # Custom exceptions
│   └── __main__.py                  # `python -m agenthatch` entry
├── agenthatch-core/                 # Universal agent runtime (independent)
│   └── src/agenthatch_core/
│       ├── bricks/                  # Plan layer, sandbox, memory, guards, workflow
│       ├── llm/                     # LLM client, Anthropic adapter, types
│       ├── loop/                    # Agent conversation loop, token counter
│       ├── mcp/                     # MCP client and configuration
│       ├── tools/                   # Tool bus, marshal, MCP loader
│       ├── context/                 # Context compaction and management
│       ├── offload/                 # Checkpoint and state persistence
│       ├── output/                  # Output sanitization
│       ├── sandbox/                 # Sandbox executors
│       ├── agent.py                 # Base Agent class
│       ├── config.py                # Runtime configuration
│       ├── state.py                 # Agent state management
│       ├── hooks.py                 # Lifecycle hooks
│       └── types.py                 # Core type definitions
└── tests/                           # pytest suite (157 tests) with SKILL.md fixtures
```

Both `src/agenthatch` and `agenthatch-core/src/agenthatch_core` are bundled into
one wheel at build time. There is no separate pip package for the core — it
lives alongside the CLI in the same install.

---

## Development workflow

### Make your changes

Work on a branch:

```bash
git checkout -b your-feature
```

### Run the quality gate

One command runs everything:

```bash
hatch run quality:check
```

This executes, in order:

1. `ruff check src/` — lint and import ordering
2. `mypy --strict src/agenthatch` — static type checking
3. `pytest` — full test suite

If the gate passes, your change is ready for review.

You can also run each step individually:

```bash
ruff check src/
mypy --strict src/agenthatch
pytest
pytest --cov          # with coverage report
pytest tests/test_config.py -v   # a single test file
```

### Code style

- Line length: 100 characters
- Target: Python 3.11
- Ruff enforces E, W, F, I, B, C4, UP rules
- Mypy runs in strict mode

---

## Issues

Found a bug or have an idea?

- Search [existing issues](https://github.com/agenthatch/agenthatch/issues) first
- Include a clear description and steps to reproduce
- Exact steps with terminal output help a lot

---

## Pull requests

- Link your PR to the issue it addresses
- Make sure `hatch run quality:check` passes locally
- Keep changes focused — one concern per PR
- No PR is too small. Docs, typos, better error messages — everything counts.

---

## Community

<!-- TODO: add Discord and other channels -->

[GitHub Discussions](https://github.com/agenthatch/agenthatch/discussions)
is the primary community space.

AI-assisted contributions are welcome. Run the quality gate
(`hatch run quality:check`) before submitting — that's all that matters.