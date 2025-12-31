"""Shared helper functions for RAG agents.

This module contains utility functions used by both VanillaRAGAgent and
MultiQueryRAGAgent, including retrieval, tracing, and common operations.
"""

from typing import Any

from agents import set_trace_processors  # pyrefly: ignore
from langsmith import get_current_run_tree
from langsmith.wrappers import OpenAIAgentsTracingProcessor

from src.agents.models import RetrievedChunk
from src.config import settings
from src.retrieval import (
    FullTextSearchRetriever,
    HybridRetriever,
    VectorSimilarityRetriever,
)

# Track whether tracing has been initialized (singleton pattern)
_tracing_initialized = False
_tracing_processor: OpenAIAgentsTracingProcessor | None = None


def retrieve_chunks(
    query: str,
    retrieval_params: dict[str, Any],
) -> list[RetrievedChunk]:
    """Retrieve relevant chunks using the retrieval system directly.

    Args:
        query: Search query
        retrieval_params: Retrieval configuration with keys:
            - mode: "fts", "vector", or "hybrid"
            - operator: "and" or "or"
            - fts_candidates: int (for hybrid mode)
            - max_returned: int
            - filters: Optional dict with metadata filters

    Returns:
        List of RetrievedChunk objects
    """
    mode = retrieval_params.get("mode", "hybrid")
    max_returned = retrieval_params.get("max_returned", 10)
    operator = retrieval_params.get("operator", "or")
    fts_candidates = retrieval_params.get("fts_candidates", 100)
    filters = retrieval_params.get("filters")

    # Select retriever based on mode (mode is always a string from API layer)
    if mode == "fts":
        retriever = FullTextSearchRetriever()
        result = retriever.retrieve(
            query=query,
            n=max_returned,
            filters=filters,
            operator=operator,
        )
    elif mode == "vector":
        retriever = VectorSimilarityRetriever()
        result = retriever.retrieve(
            query=query,
            n=max_returned,
            filters=filters,
        )
    elif mode == "hybrid":
        retriever = HybridRetriever()
        result = retriever.retrieve(
            query=query,
            n=max_returned,
            filters=filters,
            fts_candidates=fts_candidates,
            operator=operator,
        )
    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'fts', 'vector', or 'hybrid'.")

    return [
        RetrievedChunk(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            text=chunk.text,
            score=chunk.score,
            metadata=chunk.metadata,
            ord=chunk.ord,
        )
        for chunk in result.chunks
    ]


def initialize_tracing() -> None:
    """Initialize LangSmith tracing if configured.

    Sets up the OpenAIAgentsTracingProcessor to capture agent execution
    traces in LangSmith. Only initializes once, and only if LANGSMITH_API_KEY
    is set and tracing is enabled.
    """
    global _tracing_initialized, _tracing_processor

    if _tracing_initialized:
        return

    if settings.langsmith_api_key and settings.langsmith_tracing:
        _tracing_processor = OpenAIAgentsTracingProcessor(
            project_name=settings.langsmith_project,
        )
        set_trace_processors([_tracing_processor])

    _tracing_initialized = True


def flush_traces() -> None:
    """Flush any pending traces to LangSmith.

    Call this to ensure traces are sent before process exit or between tests.
    """
    if _tracing_processor is not None:
        _tracing_processor.force_flush()  # pyrefly: ignore


def shutdown_tracing() -> None:
    """Shutdown tracing and flush remaining traces.

    Call this during application shutdown to ensure all traces are sent.
    """
    if _tracing_processor is not None:
        _tracing_processor.shutdown()  # pyrefly: ignore


def get_trace_id() -> str | None:
    """Get the current LangSmith trace ID if available.

    Returns:
        The trace ID string if tracing is active, None otherwise.
    """
    try:
        run_tree = get_current_run_tree()
        if run_tree:
            return str(run_tree.id)
    except Exception:
        pass
    return None
