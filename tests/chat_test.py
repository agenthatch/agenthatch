# ruff: noqa: E501
#!/usr/bin/env python3
"""Quick chat test for a single hatched skill. Not a pytest test.
Usage: python chat_test.py <skill-name> [output-dir]
"""
import os
import sys
from pathlib import Path


def main() -> None:
    PROJECT = "/Users/didi/agenthatch_developer/project/agenthatch"
    sys.path.insert(0, f"{PROJECT}/src")
    sys.path.insert(0, f"{PROJECT}/agenthatch-core/src")

    from agenthatch.agent.runtime import SkillAgent
    from agenthatch.config import Config

    name = sys.argv[1] if len(sys.argv) > 1 else "pdf"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/agenthatch_20_test"
    ahs_path = Path(f"{output_dir}/{name}/agenthatch.yaml")

    print(f"Testing: {name} at {ahs_path}", flush=True)
    
    # Load config to get provider and default model
    config = Config.load()
    provider = config.get("agenthatch", {}).get("default", "openai")
    
    # Resolve API key from config or env
    provider_config = config.get("providers", {}).get(provider, {})
    if "." in provider:
        # custom.glm -> providers.custom.glm
        parts = provider.split(".", 1)
        provider_config = config.get("providers", {}).get(parts[0], {}).get(parts[1], {})
    
    model = provider_config.get("default_model", "glm-5.1-external")
    api_key = os.environ.get("AGENTHATCH_LLM_API_KEY", "") or provider_config.get("api_key", "")
    
    # Fallback: try provider-specific env var
    if not api_key:
        env_var = f"{provider.upper().replace('.', '_')}_API_KEY"
        api_key = os.environ.get(env_var, "")
    
    print(f"Provider: {provider}, Model: {model}, API key: {'SET' if api_key else 'NOT SET'}", flush=True)

    if not api_key:
        print("ERROR: No API key available. Chat test cannot proceed.", flush=True)
        sys.exit(1)

    agent = SkillAgent.from_ahspec(ahs_path, provider=provider, model=model, api_key=api_key)
    print("Agent loaded, chatting...", flush=True)

    resp = agent.chat("Hello! What can you help me with? Please introduce yourself briefly.")
    print("RESPONSE:", resp[:500])


if __name__ == "__main__":
    main()
