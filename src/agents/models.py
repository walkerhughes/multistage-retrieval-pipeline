"""Shared models and protocols for RAG agents."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class AgentType(str, Enum):
    """Available agent types for RAG generation."""

    VANILLA = "vanilla"
    MULTI_QUERY = "multi-query"


@dataclass
class RetrievedChunk:
    """A chunk retrieved for context in generation."""

    chunk_id: int
    doc_id: int
    text: str
    score: float
    metadata: dict[str, Any]
    ord: int


@dataclass
class AgentResponse:
    """Response from a RAG agent with full metadata.

    This structure matches the future AgentResponse from Issue #12,
    enabling seamless migration when proper agent classes are implemented.
    """

    answer: str
    trace_id: str | None
    latency_ms: float
    retrieved_chunks: list[RetrievedChunk]
    model_used: str
    tokens_used: dict[str, int] = field(default_factory=dict)


@runtime_checkable
class AgentProtocol(Protocol):
    """Protocol defining the interface for RAG agents.

    Both VanillaRAGAgent and MultiQueryRAGAgent must implement this interface.
    Using Protocol (structural subtyping) rather than ABC for flexibility.
    """

    async def generate(
        self,
        question: str,
        retrieval_params: dict[str, Any],
    ) -> AgentResponse:
        """Generate an answer to the question using RAG.

        Args:
            question: User question to answer
            retrieval_params: Dict with retrieval configuration:
                - mode: "fts", "vector", or "hybrid"
                - operator: "and" or "or"
                - fts_candidates: int (for hybrid mode)
                - max_returned: int
                - filters: Optional dict with metadata filters

        Returns:
            AgentResponse with answer and metadata
        """
        ...
