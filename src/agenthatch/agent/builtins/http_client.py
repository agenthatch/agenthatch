"""HTTP client builtin capability."""

from agenthatch.agent.builtins import BUILTIN_REGISTRY, BuiltinCapability


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

    def execute(
        self,
        method: str = "GET",
        url: str = "",
        headers: dict | None = None,
        body: str | None = None,
    ) -> str:
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
