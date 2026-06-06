"""API Template executor (agenthatch-core).

Executes API templates (curl-like HTTP requests) as tools.
"""

from __future__ import annotations

import json as _json
import logging
import os
from collections.abc import Callable
from typing import Any
from urllib import request, parse

logger = logging.getLogger(__name__)


class APITemplateExecutor:
    """Execute an API template as a tool call."""

    def __init__(self, curl: str, auth_env_var: str | None = None):
        self._curl = curl
        self._auth_env_var = auth_env_var

    def execute(self, **kwargs: Any) -> str:
        """Execute the API call with given parameters."""
        # Simple implementation: parse curl and execute
        try:
            url = self._extract_url()
            method = self._extract_method()
            headers = self._extract_headers()

            if self._auth_env_var:
                token = os.environ.get(self._auth_env_var, "")
                if token:
                    headers["Authorization"] = f"Bearer {token}"

            req_url = url
            if method == "GET" and kwargs:
                qs = parse.urlencode(kwargs)
                req_url = f"{url}?{qs}"

            req = request.Request(req_url, method=method, headers=headers)
            if method in ("POST", "PUT", "PATCH") and kwargs:
                data = _json.dumps(kwargs).encode("utf-8")
                req = request.Request(
                    req_url, data=data, method=method, headers=headers
                )
                req.add_header("Content-Type", "application/json")

            with request.urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8")
        except Exception as e:
            return f"API call failed: {e}"

    def _extract_url(self) -> str:
        """Extract URL from curl command."""
        import re
        m = re.search(r"['\"]?(https?://[^\s'\"]+)['\"]?", self._curl)
        return m.group(1) if m else ""

    def _extract_method(self) -> str:
        """Extract HTTP method from curl command."""
        if "-X" in self._curl:
            import re
            m = re.search(r"-X\s+(\w+)", self._curl)
            return m.group(1).upper() if m else "GET"
        if "--data" in self._curl or "-d" in self._curl:
            return "POST"
        return "GET"

    def _extract_headers(self) -> dict[str, str]:
        """Extract headers from curl command."""
        import re
        headers: dict[str, str] = {}
        for m in re.finditer(r"-H\s+['\"]([^'\"]+)['\"]", self._curl):
            parts = m.group(1).split(":", 1)
            if len(parts) == 2:
                headers[parts[0].strip()] = parts[1].strip()
        return headers