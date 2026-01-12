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
    python -m evals.tool_params.runner --category speaker_filter
"""

# Import dataset and metrics modules (no external dependencies beyond stdlib)
from evals.tool_params.dataset import (
    EVAL_CASES,
    EvalCase,
    ExpectedFilters,
    ToolParamsDataset,
)

# Metrics can be imported without heavy dependencies
# Runner imports are deferred to avoid requiring all dependencies at import time
# Import directly: from evals.tool_params.runner import ToolParamsHarness, ToolParamsEvalResult

__all__ = [
    # Dataset
    "EVAL_CASES",
    "EvalCase",
    "ExpectedFilters",
    "ToolParamsDataset",
]
