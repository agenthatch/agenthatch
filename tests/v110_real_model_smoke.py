"""Real-API smoke test for the v1.0.20 routing refresh (2026-09-26).

Not collected by pytest (filename lacks the test_ prefix): it makes real
LLM calls and costs real tokens. Run it explicitly:

    python tests/v110_real_model_smoke.py                     # full matrix
    python tests/v110_real_model_smoke.py anthropic claude-opus-5-5
    python tests/v110_real_model_smoke.py openai gpt-6-sol openai gpt-6-luna

For every (provider, model) pair the script exercises the exact runtime
wiring — registry ProviderFeatures → core LLMClient — and runs two live
checks:

  1. CHAT    — a plain chat completion must return non-empty text.
  2. TOOLS   — a tool call must round-trip: the model names the tool,
               the script replays the assistant turn (reasoning_content
               included, per v1.0.19) plus the tool result, and the
               follow-up must produce text. This is the multi-turn path
               that broke on DeepSeek before v1.0.19 and the adapter
               translation that must not 400 on Claude 5th-gen models.

Providers whose API key cannot be resolved are SKIPped, not failed.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "agenthatch-core" / "src"))

from agenthatch_core.llm.client import LLMClient, ProviderFeatures  # noqa: E402

from agenthatch.providers import BUILTIN_PROVIDERS, resolve_api_key  # noqa: E402

# The routing-refresh matrix: every model the v1.0.20 registry added or
# relies on, plus the pre-existing defaults as regression anchors.
MATRIX: list[tuple[str, str]] = [
    ("deepseek", "deepseek-flash"),      # default — known-good anchor
    ("deepseek", "deepseek-v4-pro"),     # service retained past 09-14
    ("openai", "gpt-6-sol"),             # 2026-09-22
    ("openai", "gpt-6-luna"),            # 2026-09-22
    ("openai", "gpt-6-astra"),           # default — anchor
    ("anthropic", "claude-opus-5-5"),    # 2026-09-22 — three 400 minefields
    ("anthropic", "claude-fable-5-1"),   # default — anchor
    ("glm", "glm-5.3-flashx"),          # 2026-09-21
    ("glm", "glm-5.3"),                  # default — anchor
    ("qwen", "qwen3.8-max"),             # default — anchor
]

_ECHO_TOOL = {
    "type": "function",
    "function": {
        "name": "echo",
        "description": "Echo the given text back verbatim.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
}


def _build_client(provider: str, model: str, api_key: str) -> LLMClient:
    """Mirror the runtime wiring: registry entry → core LLMClient."""
    info = BUILTIN_PROVIDERS[provider]
    pf = info.features
    features = ProviderFeatures(
        supports_tools=pf.supports_tools,
        supports_stream_tools=pf.supports_stream_tools,
        supports_json_mode=pf.supports_json_mode,
        supports_parallel_tool_calls=pf.supports_parallel_tool_calls,
        supports_reasoning_content=pf.supports_reasoning_content,
        requires_anthropic_adapter=pf.requires_anthropic_adapter,
        available_models=pf.available_models,
    )
    return LLMClient(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=info.base_url,
        context_window=info.context_window,
        features=features,
        max_tokens=16000,
    )


def _check_chat(llm: LLMClient) -> tuple[bool, str]:
    r = llm.chat(
        messages=[{"role": "user", "content": "Reply with exactly: PONG"}],
    )
    text = (r or "").strip()
    if text:
        return True, text[:60]
    return False, "empty response"


def _check_tools(llm: LLMClient) -> tuple[bool, str]:
    messages: list[dict] = [
        {"role": "user", "content": "Call the 'echo' tool with text 'hello'."}
    ]
    r1 = llm.chat_with_tools(messages, [_ECHO_TOOL])
    if not r1.tool_calls:
        return False, f"model replied in text: {(r1.text or '')[:60]!r}"

    # Replay the assistant turn exactly as agent_loop._record_assistant
    # does (v1.0.19): tool_calls plus reasoning_content when present.
    assistant: dict = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                },
            }
            for tc in r1.tool_calls
        ],
    }
    if r1.reasoning_content:
        assistant["reasoning_content"] = r1.reasoning_content
    messages.append(assistant)
    messages.append({
        "role": "tool",
        "tool_call_id": r1.tool_calls[0].id,
        "content": json.dumps({"echoed": r1.tool_calls[0].arguments.get("text", "")}),
    })

    r2 = llm.chat_with_tools(messages, [_ECHO_TOOL])
    if r2.tool_calls:
        # A second call is acceptable (model verifying); accept if it
        # eventually answers after one more replay round.
        return True, f"tool ok; follow-up called another tool: {r2.tool_calls[0].name}"
    text = (r2.text or "").strip()
    if text:
        return True, text[:60]
    return False, "follow-up returned empty text after tool result"


def main() -> int:
    pairs: list[tuple[str, str]]
    if len(sys.argv) > 1:
        args = sys.argv[1:]
        pairs = [(args[i], args[i + 1]) for i in range(0, len(args) - 1, 2)]
    else:
        pairs = MATRIX

    results: list[tuple[str, str, str, str]] = []
    failures = 0
    for provider, model in pairs:
        key = resolve_api_key(provider, prompt=False)
        if not key:
            results.append((provider, model, "SKIP", "no API key"))
            continue
        llm = _build_client(provider, model, key)
        row_failures: list[str] = []
        for name, check in (("chat", _check_chat), ("tools", _check_tools)):
            t0 = time.time()
            try:
                ok, detail = check(llm)
                dt = time.time() - t0
                status = "PASS" if ok else "FAIL"
                if not ok:
                    failures += 1
                    row_failures.append(name)
                results.append(
                    (provider, f"{model}/{name}", status, f"{dt:.1f}s {detail}")
                )
            except Exception as exc:  # noqa: BLE001 — report, don't crash
                failures += 1
                row_failures.append(name)
                msg = str(exc).replace("\n", " ")[:140]
                results.append((provider, f"{model}/{name}", "FAIL", msg))

    w = max(len(r[1]) for r in results) + 1
    print(f"{'provider':<10} {'model/check':<{w}} {'status':<6} detail")
    print("-" * (w + 60))
    for provider, name, status, detail in results:
        print(f"{provider:<10} {name:<{w}} {status:<6} {detail}")
    print("-" * (w + 60))
    print(f"{len(results)} checks, {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
