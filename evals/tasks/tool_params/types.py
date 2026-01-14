"""Shared type definitions for tool parameter evaluations.

This module contains dataclasses used across the tool_params evaluation
system. These are separated from runner.py to avoid heavy dependencies
when only type definitions are needed.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolCallCapture:
    """Captured tool call with its parameters.

    Attributes:
        tool_name: Name of the tool that was called
        query: The search query passed to the tool
        filters: Dictionary of filter parameters applied
        raw_args: Raw arguments as passed to the tool
    """

    tool_name: str
    query: str
    filters: dict[str, Any] = field(default_factory=dict)
    raw_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolParamsEvalResult:
    """Result of running a single tool parameter evaluation case.

    Attributes:
        case_id: ID of the eval case
        query: Original query
        expected_filters: Expected filter values
        actual_filters: Filters actually applied by the agent
        tool_calls: List of all tool calls made
        filter_matches: Dict mapping filter names to match status
        overall_match: True if all expected filters were correctly applied
        answer: The agent's generated answer
        latency_ms: Time taken for the evaluation
        error: Error message if evaluation failed
    """

    case_id: str
    query: str
    expected_filters: "ExpectedFilters"
    actual_filters: dict[str, Any]
    tool_calls: list[ToolCallCapture]
    filter_matches: dict[str, bool]
    overall_match: bool
    answer: str = ""
    latency_ms: float = 0.0
    error: Optional[str] = None


# Import ExpectedFilters here to resolve forward reference
# This creates a slight circular dependency at type level but is safe
from evals.tasks.tool_params.dataset import ExpectedFilters as ExpectedFilters
