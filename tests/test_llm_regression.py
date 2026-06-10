"""Regression tests for v0.7.7 bug fixes in LLM client.

C4: _retry() must retry on timeout/connection errors
H1: base_url fallback warning for non-openai providers
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from agenthatch_core.llm.client import LLMClient


class TestRetryTimeout:
    """C4 fix: _retry must retry on timeout and connection errors."""

    @pytest.fixture
    def client(self):
        """Create LLMClient with mocked OpenAI."""
        with patch("openai.OpenAI"):  # patched at top-level, used inside __init__
            return LLMClient(
                provider="openai",
                model="gpt-4o",
                api_key="sk-test",
                base_url="https://api.openai.com/v1",
            )

    def test_retry_on_connection_timeout(self, client):
        """A ConnectTimeout should be retried, not immediately re-raised."""
        import httpx

        call_count = 0

        def _failing_fn(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise httpx.ConnectTimeout("connection timed out")

        with pytest.raises(httpx.ConnectTimeout):
            client._retry(_failing_fn, max_retries=2)

        assert call_count == 3, (
            f"C4 regression: ConnectTimeout should be retried, "
            f"got {call_count} calls (expected 3)"
        )

    def test_retry_on_read_timeout(self, client):
        """A ReadTimeout should be retried, not immediately re-raised."""
        import httpx

        call_count = 0

        def _failing_fn(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise httpx.ReadTimeout("read timed out")

        with pytest.raises(httpx.ReadTimeout):
            client._retry(_failing_fn, max_retries=1)

        assert call_count == 2, (
            f"C4 regression: ReadTimeout should be retried, "
            f"got {call_count} calls (expected 2)"
        )

    def test_retry_on_generic_timeout_name(self, client):
        """Exceptions with 'timeout' in name should also be retried."""

        class CustomTimeoutError(Exception):
            pass

        call_count = 0

        def _failing_fn(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise CustomTimeoutError("custom timeout")

        with pytest.raises(CustomTimeoutError):
            client._retry(_failing_fn, max_retries=1)

        assert call_count == 2, (
            f"C4 regression: CustomTimeoutError should be retried "
            f"(has 'timeout' in name), got {call_count} calls"
        )

    def test_no_retry_on_value_error(self, client):
        """Non-transient errors like ValueError should NOT be retried."""

        call_count = 0

        def _failing_fn(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise ValueError("not a transient error")

        with pytest.raises(ValueError):
            client._retry(_failing_fn, max_retries=2)

        assert call_count == 1, (
            f"ValueError should NOT be retried, got {call_count} calls"
        )


class TestBaseURLFallback:
    """H1 fix: base_url fallback warning for non-openai providers."""

    @pytest.fixture
    def client_factory(self, caplog):
        """Factory fixture — creates LLMClient with mocked OpenAI."""
        import logging
        caplog.set_level(logging.WARNING)

        def _make(**kwargs):
            with patch("openai.OpenAI"):
                return LLMClient(**kwargs)

        return _make

    def test_warning_when_deepseek_no_base_url(self, client_factory, caplog):
        """LLMClient should warn when deepseek has no base_url."""
        client_factory(
            provider="deepseek",
            model="deepseek-chat",
            api_key="sk-test",
            base_url="",  # empty → triggers warning
        )

        assert any(
            "No base_url configured for provider 'deepseek'" in record.message
            for record in caplog.records
        ), "H1 regression: should warn when non-openai provider has no base_url"

    def test_no_warning_when_openai_no_base_url(self, client_factory, caplog):
        """OpenAI should NOT trigger the warning (use default URL)."""
        client_factory(
            provider="openai",
            model="gpt-4o",
            api_key="sk-test",
            base_url="",
        )

        warnings = [
            r.message for r in caplog.records
            if "No base_url configured" in r.message
        ]
        assert len(warnings) == 0, (
            "H1 regression: openai should not trigger base_url warning"
        )

    def test_no_warning_when_base_url_provided(self, client_factory, caplog):
        """No warning when base_url is explicitly set."""
        client_factory(
            provider="deepseek",
            model="deepseek-chat",
            api_key="sk-test",
            base_url="https://api.deepseek.com/v1",
        )

        warnings = [
            r.message for r in caplog.records
            if "No base_url configured" in r.message
        ]
        assert len(warnings) == 0, (
            "H1 regression: should not warn when base_url is provided"
        )
