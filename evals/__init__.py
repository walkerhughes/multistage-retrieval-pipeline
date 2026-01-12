"""Evaluation infrastructure for retrieval system.

This package provides evaluation tools organized by eval type:
- retrieval/: Retrieval quality evaluation against ground truth chunks

Use the harness to run evaluations:
    python -m evals.harness --eval-type retrieval

Or run specific eval modules directly:
    python -m evals.retrieval.runner
"""
