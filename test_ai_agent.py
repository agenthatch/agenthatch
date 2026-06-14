"""Multi-turn conversation test for AI-generated web-fetcher agent."""
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
print("AI-GENERATED AGENT TEST")
print("=" * 60)

r1 = agent.chat("Fetch https://example.com")
print(f"\n--- Turn 1 ---\n{r1}")

r2 = agent.chat("What was the page title?")
print(f"\n--- Turn 2 ---\n{r2}")

r3 = agent.chat("Now fetch https://httpbin.org/json and summarize it")
print(f"\n--- Turn 3 ---\n{r3}")

r4 = agent.chat("Can you try to fetch https://this-does-not-exist-12345.com?")
print(f"\n--- Turn 4 (error handling) ---\n{r4}")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)