"""Evaluation infrastructure for retrieval system.

This package provides evaluation tools organized by eval type:
- retrieval/: Retrieval quality evaluation against ground truth chunks
- tool_params/: Agent tool parameter extraction evaluation

Use the harness to run evaluations:
    python -m evals.harness --eval-type retrieval
    python -m evals.harness --eval-type tool-params

Or run specific eval modules directly:
    python -m evals.retrieval.runner
    python -m evals.tool_params.runner
"""
