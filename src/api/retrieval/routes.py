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
    """
    Retrieve chunks using the specified retrieval mode.

    Supports:
    - FTS: Full-Text Search with ts_rank scoring
    - Vector: Semantic search using OpenAI embeddings + pgvector
    - Hybrid: Combined FTS + Vector (coming soon)
    - Metadata filters (date ranges, doc_type, source)
    - Top N results

    Returns ranked chunks with timing information.
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
                fts_candidates=request.fts_candidates,
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
    q: str = Query(..., description="Query string"),
    mode: str = Query("fts", description="Retrieval mode: 'fts', 'vector', or 'hybrid'"),
    operator: Literal["and", "or"] = Query("or", description="FTS operator: 'or' (broad) or 'and' (strict)"),
    fts_candidates: int = Query(100, description="FTS candidates for hybrid mode", ge=1, le=500),
):
    """
    Benchmark retrieval performance.

    Returns:
    - Query execution time
    - Number of results
    - EXPLAIN ANALYZE output (index usage, scan types, etc.)

    Used to verify index efficiency and detect performance regressions.
    Supports FTS, Vector, and Hybrid retrieval benchmarking.
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
            result = retriever.retrieve(query=q, n=50, fts_candidates=fts_candidates, operator=operator)
            query_time_ms = timer.stop()

            # Get EXPLAIN output
            explain_output = retriever.explain_query(query=q, fts_candidates=fts_candidates, operator=operator)

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
