"""Vanilla RAG Agent implementation.

This is a temporary wrapper around the existing rag() function that provides
the AgentProtocol interface. When Issue #12 is implemented, this will be
replaced with a proper class-based agent with full instrumentation.
"""

from typing import Any

import httpx

from src.agents.models import AgentResponse, RetrievedChunk
from src.config import settings
from src.utils.timing import Timer


async def _retrieve_chunks(
    query: str,
    retrieval_params: dict[str, Any],
) -> list[RetrievedChunk]:
    """Retrieve relevant chunks from the retrieval API.

    Args:
        query: Search query
        retrieval_params: Retrieval configuration

    Returns:
        List of RetrievedChunk objects
    """
    async with httpx.AsyncClient(base_url=str(settings.api_base_url)) as client:
        payload: dict[str, Any] = {
            "query": query,
            "max_returned": retrieval_params.get("max_returned", 10),
            "mode": retrieval_params.get("mode", "hybrid"),
            "operator": retrieval_params.get("operator", "or"),
            "fts_candidates": retrieval_params.get("fts_candidates", 100),
        }

        if retrieval_params.get("filters"):
            payload["filters"] = retrieval_params["filters"]

        resp = await client.post("api/retrieval/query", json=payload)
        resp.raise_for_status()
        data = resp.json()

    return [
        RetrievedChunk(
            chunk_id=chunk["chunk_id"],
            doc_id=chunk["doc_id"],
            text=chunk["text"],
            score=chunk["score"],
            metadata=chunk["metadata"],
            ord=chunk["ord"],
        )
        for chunk in data["chunks"]
    ]


def _build_context(chunks: list[RetrievedChunk]) -> str:
    """Build context string from retrieved chunks.

    Args:
        chunks: List of retrieved chunks

    Returns:
        Formatted context string for the LLM
    """
    context = ""
    for chunk in chunks:
        title = chunk.metadata.get("title", "Unknown")
        context += f"""
        Title: {title}
        Text Quotation: {chunk.text}
        \n
        """
    return context


class VanillaRAGAgent:
    """Vanilla RAG agent using single-query retrieval.

    This agent:
    1. Retrieves relevant chunks using the retrieval API
    2. Builds context from retrieved chunks
    3. Generates an answer using OpenAI

    Implements AgentProtocol for compatibility with the agent factory.
    """

    async def generate(
        self,
        question: str,
        retrieval_params: dict[str, Any],
    ) -> AgentResponse:
        """Generate an answer using vanilla RAG pipeline.

        Args:
            question: User question to answer
            retrieval_params: Retrieval configuration with keys:
                - mode: "fts", "vector", or "hybrid"
                - operator: "and" or "or"
                - fts_candidates: int (for hybrid mode)
                - max_returned: int
                - filters: Optional dict with metadata filters

        Returns:
            AgentResponse with answer, trace_id, latency, chunks, and token usage
        """
        timer = Timer()
        timer.start()

        # Retrieve chunks
        chunks = await _retrieve_chunks(question, retrieval_params)

        # Build context from chunks
        context = _build_context(chunks)

        # Build messages for LLM
        system_message = (
            "Answer the user's question using only the provided information below:\n"
            + context
        )

        # Generate answer using OpenAI
        response = settings.client.chat.completions.create(
            model=settings.chat_model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": question},
            ],
        )

        # Extract answer and token usage
        answer = str(response.choices[0].message.content)
        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0
        total_tokens = response.usage.total_tokens if response.usage else 0

        tokens_used = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

        latency_ms = timer.stop()

        return AgentResponse(
            answer=answer,
            trace_id=None,  # LangSmith tracing to be implemented in a future PR
            latency_ms=latency_ms,
            retrieved_chunks=chunks,
            model_used=settings.chat_model,
            tokens_used=tokens_used,
        )
