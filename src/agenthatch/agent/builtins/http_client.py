"""HTTP client builtin capability."""

import logging
from typing import Any

from agenthatch.agent.builtins import BUILTIN_REGISTRY, BuiltinCapability, with_enriched_errors

logger = logging.getLogger(__name__)


class HttpClientCap(BuiltinCapability):
    name = "http_client"
    cap_type = "transport"
    description = "Make HTTP requests to external APIs"
    schema = {
        "type": "object",
        "properties": {
            "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE"]},
            "url": {"type": "string", "description": "Full URL including query params"},
            "headers": {"type": "object", "description": "HTTP headers"},
            "body": {"type": "string", "description": "Request body (JSON string)"},
        },
        "required": ["method", "url"],
    }

    @with_enriched_errors
    def execute(
        self,
        method: str = "GET",
        url: str = "",
        headers: dict[str, Any] | None = None,
        body: str | None = None,
        **kwargs: Any,
    ) -> str:
        if kwargs:
            logger.warning(
                "%s received unknown parameters: %s (ignored)",
                self.__class__.__name__, list(kwargs.keys()),
            )
        import httpx
        try:
            r = httpx.request(
                method, url,
                headers=headers or {},
                content=body,
                timeout=30.0,
            )
            return r.text[:5000]
        except Exception as e:
            return f"Error: {e}"


BUILTIN_REGISTRY["http_client"] = HttpClientCap
