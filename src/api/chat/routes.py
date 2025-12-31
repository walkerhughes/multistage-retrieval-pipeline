"""Chat completion API routes."""

from typing import Any

from fastapi import APIRouter, HTTPException

from src.agents.factory import get_agent
from src.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    RetrievedChunkResponse,
)

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/completion", response_model=ChatCompletionResponse)
async def chat_completion(request: ChatCompletionRequest):
    """Generate an answer using the RAG pipeline with agent selection.

    This endpoint retrieves relevant chunks from ingested documents and uses
    an LLM to generate a grounded answer. Choose an agent type and configure
    retrieval parameters for evaluation purposes.

    **Agent Types:**
    - **vanilla**: Single-query RAG - retrieves chunks for the original question
      and generates an answer. Fast and simple. (Default)
    - **multi-query**: Query decomposition RAG - breaks the question into
      2-5 MECE sub-queries, retrieves for each in parallel, deduplicates with
      score boosting, and synthesizes a comprehensive answer.

    **Retrieval Modes:**
    - **fts**: Fast keyword search (~10-50ms)
    - **vector**: Semantic search with embeddings (~50-200ms)
    - **hybrid**: Keyword search + semantic reranking (default, balanced)

    **Returns:**
    - Generated answer grounded in retrieved chunks
    - LangSmith trace ID for debugging
    - Retrieved chunks for verification
    - Timing and token usage metrics
    """
    try:
        # Get the appropriate agent
        agent = get_agent(request.agent)

        # Build retrieval params from request
        retrieval_params: dict[str, Any] = {
            "mode": request.mode.value,
            "operator": request.operator,
            "fts_candidates": request.fts_candidates,
            "max_returned": request.max_returned,
        }

        if request.filters:
            # Use mode="json" to serialize datetime objects to ISO strings
            retrieval_params["filters"] = request.filters.model_dump(mode="json", exclude_none=True)

        # Generate answer
        result = await agent.generate(request.question, retrieval_params)

        # Convert to response schema
        return ChatCompletionResponse(
            answer=result.answer,
            trace_id=result.trace_id,
            latency_ms=result.latency_ms,
            retrieved_chunks=[
                RetrievedChunkResponse(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    score=chunk.score,
                    metadata=chunk.metadata,
                    ord=chunk.ord,
                )
                for chunk in result.retrieved_chunks
            ],
            model_used=result.model_used,
            tokens_used=result.tokens_used,
            # Multi-query specific fields (empty for vanilla agent)
            sub_queries=result.sub_queries,
            chunks_per_subquery=result.chunks_per_subquery,
            deduplication_stats=result.deduplication_stats,
        )

    except NotImplementedError as e:
        # Multi-query agent not yet implemented
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        # Invalid agent type
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat completion failed: {str(e)}")
