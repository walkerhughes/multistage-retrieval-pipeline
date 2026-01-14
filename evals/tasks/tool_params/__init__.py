"""Tool parameter extraction evaluation module.

This module evaluates whether agents correctly extract and apply filter
parameters (speaker, date, source, etc.) from natural language queries.

Components:
- dataset.py: Evaluation test cases
- metrics.py: Precision/recall/F1 metrics for filter extraction
- runner.py: Harness for running evaluations

Usage:
    # Run via unified harness
    python -m evals.harness --eval-type tool-params

    # Run directly
    python -m evals.tasks.tool_params.runner --category speaker_filter
"""

# Re-export dataset types (using 'as' pattern to mark as intentional re-exports)
from evals.tasks.tool_params.dataset import (
    EVAL_CASES as EVAL_CASES,
    EvalCase as EvalCase,
    ExpectedFilters as ExpectedFilters,
    ToolParamsDataset as ToolParamsDataset,
)

# Re-export types (no heavy dependencies)
from evals.tasks.tool_params.types import (
    ToolCallCapture as ToolCallCapture,
    ToolParamsEvalResult as ToolParamsEvalResult,
)

# Runner imports are deferred to avoid requiring all dependencies at import time
# Import directly: from evals.tasks.tool_params.runner import ToolParamsHarness

__all__ = [
    # Dataset
    "EVAL_CASES",
    "EvalCase",
    "ExpectedFilters",
    "ToolParamsDataset",
    # Types
    "ToolCallCapture",
    "ToolParamsEvalResult",
]
