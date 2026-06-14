# ruff: noqa: E501
#!/usr/bin/env python3
"""Quick chat test for a single hatched skill. Not a pytest test."""
import os
import sys
from pathlib import Path


def main() -> None:
    PROJECT = "/Users/didi/agenthatch_developer/project/agenthatch"
    sys.path.insert(0, f"{PROJECT}/src")
    sys.path.insert(0, f"{PROJECT}/agenthatch-core/src")

    from agenthatch.agent.runtime import SkillAgent

    name = sys.argv[1] if len(sys.argv) > 1 else "pdf"
    ahs_path = Path(f"/tmp/agenthatch_20_test/{name}/agenthatch.yaml")

    print(f"Testing: {name} at {ahs_path}", flush=True)
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    print(f"API key: {'SET' if api_key else 'NOT SET'}", flush=True)

    agent = SkillAgent.from_ahspec(ahs_path, provider="deepseek", model="deepseek-chat", api_key=api_key)
    print("Agent loaded, chatting...", flush=True)

    resp = agent.chat("Hello! What can you help me with? Please introduce yourself briefly.")
    print("RESPONSE:", resp[:500])


if __name__ == "__main__":
    main()
