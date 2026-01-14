"""Retrieval evaluation module.

This module contains retrieval-specific evaluation logic for testing
RAG agent retrieval quality against ground truth chunk IDs.
"""

from evals.tasks.retrieval.runner import main as run_retrieval_evals

__all__ = ["run_retrieval_evals"]
