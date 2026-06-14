"""Multi-turn conversation quality test for the web-fetcher agent."""
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
print("MULTI-TURN TEST: Fetch and Compare")
print("=" * 60)

# Turn 1: Fetch example.com
r1 = agent.chat("Fetch the content from https://example.com")
print(f"\n--- Turn 1 ---\n{r1}")

# Turn 2: Ask about what was just fetched (context retention)
r2 = agent.chat("What was the title of the page you just fetched from example.com?")
print(f"\n--- Turn 2 ---\n{r2}")

# Turn 3: Fetch another URL
r3 = agent.chat("Now fetch https://httpbin.org/headers and tell me what headers were sent")
print(f"\n--- Turn 3 ---\n{r3}")

# Turn 4: Compare the two results
r4 = agent.chat("Compare the two URLs we just fetched - what was different about them?")
print(f"\n--- Turn 4 ---\n{r4}")

print("\n" + "=" * 60)
print("ERROR HANDLING TEST")
print("=" * 60)

# Test error handling
r5 = agent.chat("Try to fetch https://this-domain-does-not-exist-12345.com")
print(f"\n--- Error Test ---\n{r5}")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)