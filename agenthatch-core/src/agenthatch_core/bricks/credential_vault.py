"""CredentialVault — proxy-based credential management.

Level 0 — agents declare "I need X credential", the vault resolves
and injects credentials at the transport layer.  Agent code never
reads raw keys — it only references credential names.

Designed for the closure pattern with APITemplateExecutor:
    executor.execute = lambda **kwargs: vault.resolve("github") and ...
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CredentialEntry:
    """A stored credential."""
    name: str
    value: str = ""          # resolved value (never serialized)
    env_var: str = ""        # source environment variable
    provider: str = ""       # "env" | "keyring" | "vault" | "config"
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class CredentialVault:
    """Proxy-based credential manager.

    Usage:
        vault = CredentialVault()
        vault.register("github_token", env_var="GITHUB_TOKEN")
        token = vault.resolve("github_token")  # reads from GITHUB_TOKEN env var
    """

    entries: dict[str, CredentialEntry] = field(default_factory=dict)

    def register(
        self,
        name: str,
        env_var: str = "",
        provider: str = "env",
        metadata: dict[str, str] | None = None,
    ) -> None:
        """Register a credential requirement."""
        self.entries[name] = CredentialEntry(
            name=name,
            env_var=env_var,
            provider=provider,
            metadata=metadata or {},
        )

    def resolve(self, name: str) -> str:
        """Resolve a credential by name.

        Resolution order:
        1. If entry has pre-set value, return it
        2. Check environment variable
        3. Return empty string (caller must handle)
        """
        entry = self.entries.get(name)
        if entry is None:
            return ""

        if entry.value:
            return entry.value

        if entry.env_var:
            value = os.environ.get(entry.env_var, "")
            if value:
                return value

        return ""

    def inject_into_headers(self, headers: dict[str, str]) -> dict[str, str]:
        """Inject resolved credentials into HTTP headers.

        Each registered credential with an 'Authorization' or 'X-API-Key'
        header mapping in metadata will be resolved and injected.
        """
        result = dict(headers)
        for name, entry in self.entries.items():
            header_name = entry.metadata.get("header")
            if header_name:
                value = self.resolve(name)
                if value:
                    scheme = entry.metadata.get("scheme", "Bearer")
                    result[header_name] = f"{scheme} {value}"
        return result

    def store(self, name: str, value: str) -> None:
        """Store a credential value (in-memory only, never persisted)."""
        if name in self.entries:
            self.entries[name].value = value
        else:
            self.entries[name] = CredentialEntry(
                name=name, value=value, provider="inline"
            )

    def dump(self) -> str:
        """Serialize credential registry to TOML for secure persistence.

        Uses tomli_w.dumps() to ensure special characters (``=``, ``"``,
        newlines, etc.) in credential values are properly escaped.
        Credential values are NEVER included — only metadata is serialized.
        """
        import tomli_w

        data: dict[str, dict[str, Any]] = {}
        for name, entry in self.entries.items():
            data[name] = {
                "env_var": entry.env_var,
                "provider": entry.provider,
                "metadata": entry.metadata,
            }
        return tomli_w.dumps({"credentials": data})
