from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import BenchmarkResponse, ChunkResult, QueryRequest, QueryResponse
from src.retrieval import (
    FullTextSearchRetriever,
    HybridRetriever,
    RetrievalMode,
    VectorSimilarityRetriever,
)
from src.utils.timing import Timer

router = APIRouter(prefix="/retrieval", tags=["Retrieval"])


@router.post("/query", response_model=QueryResponse)
async def query_chunks(request: QueryRequest):
    """Retrieve relevant text chunks from ingested documents

    This endpoint searches through all ingested documents and returns the most
    relevant text chunks based on your query. Choose your retrieval strategy:

    **Retrieval Modes:**
    - **fts**: Fast keyword search using database indexes (~10-50ms). Best for
      factual queries with specific terms.
    - **vector**: Semantic search using AI embeddings. Slower but understands
      meaning. Best for conceptual queries ('explain X', 'what is Y').
    - **hybrid**: Combines both approaches - keyword search retrieves candidates,
      then semantic ranking picks the best. Balanced recall and relevance.

    **Query Operator (FTS & Hybrid modes):**
    - **or**: Returns chunks matching ANY term (default, broader results)
    - **and**: Returns chunks matching ALL terms (stricter, more specific)

    **Metadata Filtering:**
    Optional filters to narrow results by source, document type, or date range.

    **Returns:**
    - Ranked chunks with relevance scores
    - Timing breakdown for debugging
    - Query metadata
    """
    try:
        # Convert filters to dict if provided
        filters_dict = (
            request.filters.model_dump(exclude_none=True) if request.filters else None
        )

        # Execute retrieval based on mode
        if request.mode == RetrievalMode.FTS:
            retriever = FullTextSearchRetriever()
            result = retriever.retrieve(
                query=request.q,
                n=request.n,
                filters=filters_dict,
                operator=request.operator,
            )
        elif request.mode == RetrievalMode.VECTOR:
            retriever = VectorSimilarityRetriever()
            result = retriever.retrieve(
                query=request.q,
                n=request.n,
                filters=filters_dict,
            )
        elif request.mode == RetrievalMode.HYBRID:
            retriever = HybridRetriever()
            result = retriever.retrieve(
                query=request.q,
                n=request.n,
                filters=filters_dict,
                fts_candidates=request.fts_candidates_for_reranking,
                operator=request.operator,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid mode: {request.mode}. Use 'fts', 'vector', or 'hybrid'.",
            )

        # Convert to response schema
        return QueryResponse(
            chunks=[
                ChunkResult(
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    score=c.score,
                    text=c.text,
                    metadata=c.metadata,
                    ord=c.ord,
                )
                for c in result.chunks
            ],
            timing_ms=result.timing_ms,
            query_info=result.query_info,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.get("/bench", response_model=BenchmarkResponse)
async def benchmark_retrieval(
    q: str = Query(
        ...,
        description="Search query (required). Example: 'machine learning optimization'",
        examples=["what is attention in transformers?"]
    ),
    mode: str = Query(
        "fts",
        description="Retrieval mode to benchmark: 'fts' (keyword search), 'vector' (semantic search), or 'hybrid' (combined)",
        examples=["fts"]
    ),
    operator: Literal["and", "or"] = Query(
        "or",
        description="FTS query operator (applies to 'fts' and 'hybrid' modes): 'or' = match any term (default), 'and' = match all terms",
        examples=["or"]
    ),
    fts_candidates_for_reranking: int = Query(
        100,
        description="Number of initial keyword-matched chunks to rerank semantically in 'hybrid' mode (range: 1-500, default: 100)",
        ge=1,
        le=500,
        examples=[100]
    ),
):
    """Benchmark retrieval performance and verify index efficiency

    Use this endpoint to measure query performance, check index usage, and
    detect performance regressions. Returns detailed EXPLAIN ANALYZE output
    showing how the database executes your query.

    **Performance Targets:**
    - FTS: ~10-50ms (should use GIN index on tsvector)
    - Vector: ~50-200ms (includes embedding generation)
    - Hybrid: ~100-300ms (FTS + embedding + reranking)

    **What You Get:**
    - Query execution time in milliseconds
    - Number of matching chunks
    - EXPLAIN ANALYZE output showing:
      - Index usage (critical for performance)
      - Scan types and row counts
      - Planning and execution time breakdown
    """
    try:
        # Select retriever based on mode
        if mode == "fts":
            retriever = FullTextSearchRetriever()

            # Time the query
            timer = Timer()
            timer.start()
            result = retriever.retrieve(query=q, n=50, operator=operator)
            query_time_ms = timer.stop()

            # Get EXPLAIN output
            explain_output = retriever.explain_query(query=q, operator=operator)

            return BenchmarkResponse(
                query_time_ms=round(query_time_ms, 2),
                rows_returned=len(result.chunks),
                explain=explain_output,
                query=q,
            )

        elif mode == "vector":
            retriever = VectorSimilarityRetriever()

            # Time the query (includes embedding generation)
            timer = Timer()
            timer.start()
            result = retriever.retrieve(query=q, n=50)
            query_time_ms = timer.stop()

            # Get EXPLAIN output (returns tuple with embedding time)
            explain_output, embedding_ms = retriever.explain_query(query=q)

            # Include embedding time in response
            explain_with_timing = (
                f"Embedding Generation: {embedding_ms:.2f}ms\n\n{explain_output}"
            )

            return BenchmarkResponse(
                query_time_ms=round(query_time_ms, 2),
                rows_returned=len(result.chunks),
                explain=explain_with_timing,
                query=q,
            )

        elif mode == "hybrid":
            retriever = HybridRetriever()

            # Time the query (includes FTS + embedding + reranking)
            timer = Timer()
            timer.start()
            result = retriever.retrieve(query=q, n=50, fts_candidates=fts_candidates_for_reranking, operator=operator)
            query_time_ms = timer.stop()

            # Get EXPLAIN output
            explain_output = retriever.explain_query(query=q, fts_candidates=fts_candidates_for_reranking, operator=operator)

            return BenchmarkResponse(
                query_time_ms=round(query_time_ms, 2),
                rows_returned=len(result.chunks),
                explain=explain_output,
                query=q,
            )

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid mode: {mode}. Use 'fts', 'vector', or 'hybrid'.",
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")
