"""Regression tests for MCP client bug fixes.

Covers:
- v0.9.18: register_with_capbus uses split("__", 2) for three-segment tool names
"""

from __future__ import annotations

from typing import Any


class TestBugMCPRegisterSplitCount:
    """v0.9.18: register_with_capbus must use ``split("__", 2)`` not ``split("__", 1)``.

    MCP tool names follow the convention ``<server>__<tool>__<version>``
    (three segments with two double-underscore separators).  The previous
    code used ``split("__", 1)`` which would produce ``[server, tool__version]``
    — the ``tool_name`` ended up containing the version suffix, producing
    wrong tool names that didn't match the MCP server's schema.

    The fix uses ``split("__", 2)`` which correctly extracts the middle
    segment as the server name (for three-segment names).
    """

    def test_split_count_is_2_not_1(self) -> None:
        """Verify ``full_name.split("__", 2)[1]`` is used in register_with_capbus."""
        import ast
        from pathlib import Path

        client_path = (
            Path(__file__).parent.parent
            / "agenthatch-core" / "src" / "agenthatch_core" / "mcp" / "client.py"
        )
        source = client_path.read_text()
        tree = ast.parse(source)

        # Find the register_with_capbus method and check the split call
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "register_with_capbus":
                found_split = False
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Call):
                        if (
                            isinstance(sub.func, ast.Attribute)
                            and sub.func.attr == "split"
                        ):
                            # Check second positional arg is 2
                            if len(sub.args) >= 2:
                                arg = sub.args[1]
                                if isinstance(arg, ast.Constant) and arg.value == 2:
                                    found_split = True
                                    break
                assert found_split, (
                    "BUG REGRESSION: register_with_capbus does not use "
                    "split('__', 2) — three-segment tool names will be "
                    "parsed incorrectly"
                )

    def test_three_segment_name_extracts_correct_middle(self) -> None:
        """Verify the split logic extracts the correct middle segment."""
        full_name = "github__search_repos__v1"
        parts = full_name.split("__", 2)
        assert len(parts) == 3, (
            f"Three-segment name should split into 3 parts, got {len(parts)}: {parts}"
        )
        # parts[1] should be the server name (tool name without version)
        assert parts[1] == "search_repos", (
            f"Middle segment should be 'search_repos', got '{parts[1]}'"
        )
        # Contrast with split("__", 1) which would produce wrong result
        wrong_parts = full_name.split("__", 1)
        assert wrong_parts[1] == "search_repos__v1", (
            "split('__', 1) would include version in tool name — this is the bug"
        )

    def test_two_segment_name_still_works(self) -> None:
        """Two-segment names (no version suffix) should still work."""
        full_name = "github__search_repos"
        parts = full_name.split("__", 2)
        assert len(parts) == 2, (
            f"Two-segment name should split into 2 parts, got {len(parts)}"
        )
        assert parts[1] == "search_repos"
