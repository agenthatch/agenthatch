"""LLMClient — Thin wrapper over v0.2 providers + openai SDK.

All AgentHarnesses use this single client interface.
Supports OpenAI-compatible APIs (including Anthropic proxy via base_url).

Usage:
    client = LLMClient(provider_name="openai")
    response = client.chat(messages=[{"role": "user", "content": "Hello"}])
    result = client.chat_structured(messages=msgs, response_model=MyPydanticModel)
"""

from __future__ import annotations

from typing import Any

from agenthatch.exceptions import ApiKeyError
from agenthatch.providers import get_default_provider, get_provider, resolve_api_key


class LLMClient:
    """Unified LLM call interface wrapping v0.2 provider management.

    All AgentHarnesses use this client for both simple chat and
    structured (Instructor) output calls.
    """

    def __init__(self, provider_name: str | None = None, model: str | None = None):
        """Initialize LLM client for a provider.

        Args:
            provider_name: Provider name (openai/anthropic/deepseek/ollama/custom.<name>).
                           If None, uses the default provider from config.
            model: Model name. If None, uses provider's default_model.
        """
        name = provider_name or get_default_provider()
        self._info = get_provider(name)
        api_key = resolve_api_key(name)
        if not api_key:
            raise ApiKeyError(f"No API key resolved for provider '{name}'")

        self._provider_name = name
        self._model = model or self._info.default_model

        import openai

        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=self._info.base_url,
            timeout=120.0,
        )

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model(self) -> str:
        return self._model

    # ── Simple chat ──────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Simple chat completion. Returns response text."""
        response = self._client.chat.completions.create(
            model=model or self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    # ── Structured output (Instructor pattern) ───────────────────────

    def chat_structured(
        self,
        messages: list[dict[str, Any]],
        response_model: type,
        model: str | None = None,
        max_retries: int = 2,
    ) -> Any:
        """Structured output via Instructor (LLM → Pydantic).

        Wraps instructor.from_openai() with retry loop.
        Returns validated Pydantic model instance.
        """
        import instructor

        # Mode.JSON required: glm-5-external in TOOLS mode serializes
        # nested Pydantic objects as JSON strings instead of native dicts,
        # causing validation errors like "Input should be an object".
        client = instructor.from_openai(self._client, mode=instructor.Mode.JSON)
        return client.chat.completions.create(
            model=model or self._model,
            messages=messages,  # type: ignore[arg-type]
            response_model=response_model,
            max_retries=max_retries,
        )
