"""Shared models and enums for retrieval system."""

from dataclasses import dataclass
from enum import Enum


class RetrievalMode(str, Enum):
    """Retrieval mode options."""

    FTS = "fts"  # Full-Text Search only
    VECTOR = "vector"  # Vector similarity only
    HYBRID = "hybrid"  # Combined FTS + Vector


@dataclass
class RetrievalResult:
    """Single chunk retrieval result."""

    chunk_id: int
    doc_id: int
    text: str
    score: float
    metadata: dict
    ord: int


@dataclass
class RetrievalResponse:
    """Complete retrieval response with timing."""

    chunks: list[RetrievalResult]
    timing_ms: dict
    query_info: dict
