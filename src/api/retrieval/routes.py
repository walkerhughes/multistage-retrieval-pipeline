from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import (
    BenchmarkResponse,
    ChunkResult,
    ExpandRequest,
    ExpandResponse,
    QAPair,
    QAPairsRequest,
    QAPairsResponse,
    QueryRequest,
    QueryResponse,
    TurnData,
)
from src.database.connection import execute_query
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
                query=request.query,
                n=request.max_returned,
                filters=filters_dict,
                operator=request.operator,
            )
        elif request.mode == RetrievalMode.VECTOR:
            retriever = VectorSimilarityRetriever()
            result = retriever.retrieve(
                query=request.query,
                n=request.max_returned,
                filters=filters_dict,
            )
        elif request.mode == RetrievalMode.HYBRID:
            retriever = HybridRetriever()
            result = retriever.retrieve(
                query=request.query,
                n=request.max_returned,
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


@router.post("/expand", response_model=ExpandResponse)
async def expand_chunks(request: ExpandRequest):
    """Expand chunk IDs to retrieve full speaker turn context

    Given a list of chunk IDs (e.g., from retrieval results), this endpoint
    fetches the complete speaker turn for each chunk. This is useful for:

    - Getting full context around retrieved chunk snippets
    - Preserving speaker attribution
    - Understanding complete utterances that were split across chunks

    **Deduplication:** If multiple chunks belong to the same turn, that turn
    appears only once in the response.

    **Returns:**
    - List of turn data with speaker, full text, timestamps, section titles
    - Total count of unique turns
    - Query execution time
    """
    try:
        timer = Timer()
        timer.start()

        # Query: Get distinct turns for the provided chunk IDs
        query = """
            SELECT DISTINCT
                t.id AS turn_id,
                t.doc_id,
                t.ord,
                t.speaker,
                t.text AS full_text,
                t.start_time_seconds,
                t.section_title,
                t.token_count
            FROM turns t
            INNER JOIN chunks c ON c.turn_id = t.id
            WHERE c.id = ANY(%(chunk_ids)s)
            ORDER BY t.doc_id, t.ord
        """

        results = execute_query(query, {"chunk_ids": request.chunk_ids})

        query_time_ms = timer.stop()

        # Convert to TurnData objects
        turns = [
            TurnData(
                turn_id=row["turn_id"],
                doc_id=row["doc_id"],
                ord=row["ord"],
                speaker=row["speaker"],
                full_text=row["full_text"],
                start_time_seconds=row["start_time_seconds"],
                section_title=row["section_title"],
                token_count=row["token_count"],
            )
            for row in results
        ]

        return ExpandResponse(
            turns=turns,
            total_turns=len(turns),
            query_time_ms=round(query_time_ms, 2),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Expansion failed. Please verify the chunk IDs exist."
        )


@router.post("/qa-pairs", response_model=QAPairsResponse)
async def generate_qa_pairs(request: QAPairsRequest):
    """Generate Q&A pairs from turn IDs by pairing with previous turns

    For each provided turn ID (the "answer"), this endpoint retrieves the
    previous turn (the "question") from the same document. This creates
    natural Q&A pairs for:

    - Training question-answering models
    - Building evaluation datasets
    - Understanding conversational context

    **Flexible Speaker Pairing:** Works with any speaker combination (not
    limited to specific names like "Dwarkesh" or "Guest").

    **Ordering:** Pairs are based on the `ord` field within each document.
    The previous turn is the one with `ord = current_ord - 1`.

    **Skipped Turns:** Turns without a valid previous turn (e.g., first turn
    in a document) are skipped and counted in `skipped_turns`.

    **Returns:**
    - List of Q&A pairs (previous turn + target turn)
    - Count of pairs generated
    - Count of turns skipped (first-turn in doc, no previous)
    - Count of turn IDs not found in database
    - Query execution time
    """
    try:
        timer = Timer()
        timer.start()

        # Deduplicate input turn IDs
        unique_turn_ids = list(set(request.turn_ids))

        # Query: Get turn pairs (current + previous) for provided turn IDs
        # Also returns answer_turn_id for all found turns to calculate not_found
        query = """
            WITH target_turns AS (
                SELECT
                    t.id AS answer_turn_id,
                    t.doc_id,
                    t.ord AS answer_ord,
                    t.speaker AS answer_speaker,
                    t.text AS answer_text,
                    t.start_time_seconds AS answer_start_time,
                    t.section_title AS answer_section,
                    t.token_count AS answer_tokens
                FROM turns t
                WHERE t.id = ANY(%(turn_ids)s)
            ),
            previous_turns AS (
                SELECT
                    t.id AS question_turn_id,
                    t.doc_id,
                    t.ord AS question_ord,
                    t.speaker AS question_speaker,
                    t.text AS question_text,
                    t.start_time_seconds AS question_start_time,
                    t.section_title AS question_section,
                    t.token_count AS question_tokens
                FROM turns t
            )
            SELECT
                -- Question (previous turn)
                pt.question_turn_id,
                pt.doc_id AS question_doc_id,
                pt.question_ord,
                pt.question_speaker,
                pt.question_text,
                pt.question_start_time,
                pt.question_section,
                pt.question_tokens,
                -- Answer (target turn)
                tt.answer_turn_id,
                tt.doc_id AS answer_doc_id,
                tt.answer_ord,
                tt.answer_speaker,
                tt.answer_text,
                tt.answer_start_time,
                tt.answer_section,
                tt.answer_tokens
            FROM target_turns tt
            INNER JOIN previous_turns pt
                ON tt.doc_id = pt.doc_id
                AND tt.answer_ord = pt.question_ord + 1
            ORDER BY tt.doc_id, tt.answer_ord
        """

        results = execute_query(query, {"turn_ids": unique_turn_ids})

        query_time_ms = timer.stop()

        # Convert to QAPair objects
        pairs = [
            QAPair(
                question_turn=TurnData(
                    turn_id=row["question_turn_id"],
                    doc_id=row["question_doc_id"],
                    ord=row["question_ord"],
                    speaker=row["question_speaker"],
                    full_text=row["question_text"],
                    start_time_seconds=row["question_start_time"],
                    section_title=row["question_section"],
                    token_count=row["question_tokens"],
                ),
                answer_turn=TurnData(
                    turn_id=row["answer_turn_id"],
                    doc_id=row["answer_doc_id"],
                    ord=row["answer_ord"],
                    speaker=row["answer_speaker"],
                    full_text=row["answer_text"],
                    start_time_seconds=row["answer_start_time"],
                    section_title=row["answer_section"],
                    token_count=row["answer_tokens"],
                ),
            )
            for row in results
        ]

        # Count found turn IDs to distinguish not_found from skipped
        found_turn_ids = {row["answer_turn_id"] for row in results}

        # To properly count not_found, we need to check which turns exist
        # The pairs only include turns with previous turns, so we need separate query
        found_count_query = """
            SELECT COUNT(*) as cnt FROM turns WHERE id = ANY(%(turn_ids)s)
        """
        count_result = execute_query(found_count_query, {"turn_ids": unique_turn_ids})
        valid_turns_count = count_result[0]["cnt"] if count_result else 0

        not_found_count = len(unique_turn_ids) - valid_turns_count
        skipped_turns = valid_turns_count - len(pairs)  # Valid turns without previous turn

        return QAPairsResponse(
            pairs=pairs,
            total_pairs=len(pairs),
            skipped_turns=skipped_turns,
            not_found_count=not_found_count,
            query_time_ms=round(query_time_ms, 2),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Q&A pair generation failed. Please verify the turn IDs exist."
        )
