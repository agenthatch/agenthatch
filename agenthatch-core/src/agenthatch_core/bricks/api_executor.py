"""APITemplateExecutor — compiled API executor with retry and rate-limit.

Level 0 — compiled from API template declarations into executable
callables.  Features:
- Exponential backoff for 429/5xx responses
- Token-bucket rate limiting
- Credential proxy via vault (closure pattern)
"""

from __future__ import annotations

import json
import logging
import random
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class RateLimiter:
    """Simple token-bucket rate limiter."""
    rate: float = 10.0       # requests per second
    burst: int = 20           # max burst size
    _tokens: float = field(default=20.0, init=False)
    _last_refill: float = field(default_factory=time.monotonic, init=False)

    def acquire(self) -> bool:
        """Try to acquire one token. Returns True if allowed."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
        self._last_refill = now

        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False

    def wait(self) -> None:
        """Block until a token is available."""
        while not self.acquire():
            time.sleep(0.05)


@dataclass
class ApiTemplateConfig:
    """Compiled API template configuration."""
    name: str = ""
    url: str = ""
    method: str = "GET"
    headers: dict[str, str] = field(default_factory=dict)
    auth_env_var: str = ""
    credential_name: str = ""
    rate_limit: float = 10.0
    timeout: float = 30.0
    max_retries: int = 3

    # Retry configuration
    retry_statuses: set[int] = field(default_factory=lambda: {429, 500, 502, 503, 504})


class APITemplateExecutor:
    """Compiled API executor with retry, rate-limit, and credential proxy.

    Usage:
        executor = APITemplateExecutor.from_template(tmpl)
        result = executor.execute(city="London")
    """

    def __init__(
        self,
        config: ApiTemplateConfig,
        vault: Any = None,   # CredentialVault instance (optional, injected via closure)
    ):
        self._config = config
        self._vault = vault
        self._rate_limiter = RateLimiter(rate=config.rate_limit)

    def execute(self, **kwargs: Any) -> str:
        """Execute the API call with retry and rate limiting."""
        self._rate_limiter.wait()

        url = self._config.url
        headers = dict(self._config.headers)
        method = self._config.method.upper()

        # Inject credentials from vault
        if self._vault and self._config.credential_name:
            headers = self._vault.inject_into_headers(headers)
        elif self._config.auth_env_var:
            import os
            token = os.environ.get(self._config.auth_env_var, "")
            if token:
                headers["Authorization"] = f"Bearer {token}"

        # Build request
        if method == "GET" and kwargs:
            import urllib.parse
            qs = urllib.parse.urlencode(kwargs)
            url = f"{url}?{qs}"

        req = urllib.request.Request(url, method=method, headers=headers)

        if method in ("POST", "PUT", "PATCH") and kwargs:
            data = json.dumps(kwargs).encode("utf-8")
            req = urllib.request.Request(url, data=data, method=method, headers=headers)
            req.add_header("Content-Type", "application/json")

        return self._call_with_retry(req)

    def _call_with_retry(self, req: urllib.request.Request) -> str:
        """Execute request with exponential backoff."""
        last_error: Exception | None = None

        for attempt in range(self._config.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self._config.timeout) as resp:
                    return resp.read().decode("utf-8")
            except urllib.error.HTTPError as e:
                last_error = e
                if e.code not in self._config.retry_statuses:
                    return f"API error: {e.code} {e.reason}"
                if attempt == self._config.max_retries:
                    break
            except Exception as e:
                last_error = e
                if attempt == self._config.max_retries:
                    break

            delay = min(1.0 * (2 ** attempt), 30.0)
            delay *= random.uniform(0.75, 1.25)
            logger.debug(
                "API retry %d/%d for %s %s after %.1fs: %s",
                attempt + 1, self._config.max_retries,
                self._config.method, self._config.url, delay, last_error,
            )
            time.sleep(delay)

        return f"API call failed after {self._config.max_retries + 1} attempts: {last_error}"

    @classmethod
    def from_template(
        cls,
        tmpl: dict[str, Any],
        vault: Any = None,
    ) -> APITemplateExecutor:
        """Create executor from API template dict."""
        config = ApiTemplateConfig(
            name=tmpl.get("name", ""),
            url=tmpl.get("url", ""),
            method=tmpl.get("method", "GET"),
            headers=tmpl.get("headers", {}),
            auth_env_var=tmpl.get("auth_env_var", ""),
            credential_name=tmpl.get("credential", ""),
            rate_limit=tmpl.get("rate_limit", 10.0),
            timeout=tmpl.get("timeout", 30.0),
            max_retries=tmpl.get("max_retries", 3),
        )
        return cls(config, vault=vault)
