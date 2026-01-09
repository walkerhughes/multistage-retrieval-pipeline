"""Evaluation framework for agent tool parameter extraction.

This module provides tools for evaluating agent tool parameter extraction:

- EvalCase: A single evaluation test case
- EvalDataset: Container for evaluation test cases
- ExpectedFilters: Expected filter parameters
- EvalHarness: Harness for running evaluations
- EvalResult: Result from a single evaluation
- ToolCallCapture: Captured tool call details
- EvalMetrics: Aggregate evaluation metrics
- compute_metrics: Compute metrics from results

Usage:
    from src.evals.dataset import EvalDataset, EvalCase, ExpectedFilters
    from src.evals.metrics import compute_metrics, EvalMetrics

    # For running evaluations (requires config/OpenAI):
    from src.evals.harness import EvalHarness, EvalResult, ToolCallCapture
"""

# Import dataset and metrics modules (no external dependencies)
from src.evals.dataset import EvalCase, EvalDataset, ExpectedFilters
from src.evals.metrics import EvalMetrics, compute_metrics

# Harness imports are deferred to avoid config requirements at import time
# Import directly: from src.evals.harness import EvalHarness, EvalResult, ToolCallCapture

__all__ = [
    "EvalCase",
    "EvalDataset",
    "ExpectedFilters",
    "EvalMetrics",
    "compute_metrics",
]
