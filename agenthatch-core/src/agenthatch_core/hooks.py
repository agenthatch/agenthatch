"""Hooks manager (agenthatch-core)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class HookPoint(Enum):
    PRE_COMPACT = "pre_compact"
    POST_COMPACT = "post_compact"
    PRE_TOOL_CALL = "pre_tool_call"
    POST_TOOL_CALL = "post_tool_call"
    PRE_TURN = "pre_turn"
    POST_TURN = "post_turn"


HookCallback = Callable[[dict[str, Any]], dict[str, Any] | None]


class HooksManager:
    def __init__(self) -> None:
        self._hooks: dict[HookPoint, list[tuple[int, str, HookCallback]]] = {
            p: [] for p in HookPoint
        }

    def register(
        self, point: HookPoint, callback: HookCallback,
        priority: int = 50, name: str = "",
    ) -> None:
        self._hooks[point].append((priority, name, callback))
        self._hooks[point].sort(key=lambda x: x[0])

    def execute(self, point: HookPoint, context: dict[str, Any]) -> dict[str, Any]:
        result = context
        for _, name, callback in self._hooks[point]:
            try:
                modified = callback(result)
                if modified is not None:
                    result = modified
            except Exception as e:
                logger.warning("Hook '%s' at %s failed: %s", name, point.value, e)
        return result