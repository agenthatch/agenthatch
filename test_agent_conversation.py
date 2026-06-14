"""Test the generated web-fetcher agent with real conversation."""
import sys
sys.path.insert(0, "/Users/didi/agenthatch_developer/project/agenthatch/web-fetcher-agent/src")

from web_fetcher.agent import WebFetcher

agent = WebFetcher(runtime_config={
    "llm": {
        "provider": "deepseek",
        "model": "deepseek-v4-pro",
        "api_key": "sk-d7c914da78a649608c3cc2a55e66135c",
        "base_url": "https://api.deepseek.com",
    }
})

print("=" * 60)
print("TEST 1: Basic fetch")
print("=" * 60)
result = agent.chat("Fetch the content from https://example.com")
print(result)
print()

print("=" * 60)
print("TEST 2: What tools do you have?")
print("=" * 60)
result = agent.chat("What tools do you have available?")
print(result)
print()

print("=" * 60)
print("TEST 3: Complex request")
print("=" * 60)
result = agent.chat("Can you fetch https://httpbin.org/json and tell me what JSON data it returns?")
print(result)