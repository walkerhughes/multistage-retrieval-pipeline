"""Hybrid retrieval combining FTS first-stage with vector reranking."""

from typing import Literal, Optional

from src.database.connection import get_db_connection
from src.embeddings.service import EmbeddingService
from src.retrieval.fts import FullTextSearchRetriever
from src.retrieval.models import RetrievalResponse, RetrievalResult
from src.utils.timing import Timer


class HybridRetriever:
    """
    Hybrid retrieval: FTS first-stage (broad recall) → Vector reranking (precision).

    Two-stage approach:
    1. FTS retrieves N candidates (default: 50) for broad coverage
    2. Vector similarity reranks candidates for semantic relevance

    This aligns with: "Retrieve broadly and cheaply → reason narrowly and expensively"
    """

    def __init__(self):
        """Initialize with FTS retriever and embedding service."""
        self.fts_retriever = FullTextSearchRetriever()
        self.embedding_service = EmbeddingService()

    def retrieve(
        self,
        query: str,
        n: int = 50,
        filters: Optional[dict] = None,
        fts_candidates: int = 100,
        operator: Literal["and", "or"] = "or",
    ) -> RetrievalResponse:
        """
        Retrieve and rerank chunks using hybrid approach.

        Args:
            query: User query string
            n: Number of final results to return (after reranking)
            filters: Optional metadata filters (date ranges, doc_type, etc.)
            fts_candidates: Number of candidates to retrieve with FTS (default: 100)
            operator: FTS operator ("or" for broad, "and" for strict)

        Returns:
            RetrievalResponse with reranked chunks, timing, and query info
        """
        timer = Timer()

        # Stage 1: FTS retrieval (broad recall)
        timer.start()
        fts_results = self.fts_retriever.retrieve(
            query=query,
            n=fts_candidates,
            filters=filters,
            operator=operator,
        )
        fts_ms = timer.stop()

        # If no FTS results, return empty
        if not fts_results.chunks:
            return RetrievalResponse(
                chunks=[],
                timing_ms={
                    "fts": round(fts_ms, 2),
                    "embedding": 0.0,
                    "reranking": 0.0,
                    "total": round(fts_ms, 2),
                },
                query_info={
                    "query": query,
                    "n": n,
                    "fts_candidates": fts_candidates,
                    "results_returned": 0,
                    "filters_applied": filters or {},
                    "retrieval_mode": "hybrid",
                },
            )

        # Stage 2: Vector reranking
        timer.start()
        query_embedding = self.embedding_service.embed_text(query)
        embedding_ms = timer.stop()

        # Rerank using database-computed cosine similarity
        timer.start()
        chunk_ids = [chunk.chunk_id for chunk in fts_results.chunks]
        reranked_chunks = self._rerank_by_similarity_db(
            fts_results.chunks,
            chunk_ids,
            query_embedding,
        )
        reranking_ms = timer.stop()

        # Return top N after reranking
        top_n = reranked_chunks[:n]
        total_ms = fts_ms + embedding_ms + reranking_ms

        return RetrievalResponse(
            chunks=top_n,
            timing_ms={
                "fts": round(fts_ms, 2),
                "embedding": round(embedding_ms, 2),
                "reranking": round(reranking_ms, 2),
                "total": round(total_ms, 2),
            },
            query_info={
                "query": query,
                "n": n,
                "fts_candidates": fts_candidates,
                "results_returned": len(top_n),
                "filters_applied": filters or {},
                "retrieval_mode": "hybrid",
            },
        )

    def _rerank_by_similarity_db(
        self,
        chunks: list[RetrievalResult],
        chunk_ids: list[int],
        query_embedding: list[float],
    ) -> list[RetrievalResult]:
        """
        Rerank chunks by cosine similarity using database computation.

        More efficient than Python-based similarity since pgvector is optimized.

        Args:
            chunks: FTS candidate chunks
            chunk_ids: List of chunk IDs to rerank
            query_embedding: Query embedding vector

        Returns:
            Reranked list of chunks sorted by similarity (descending)
        """
        if not chunk_ids:
            return []

        # Convert embedding to pgvector format string '[1,2,3,...]'
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        # Build query to compute similarity for chunk IDs
        placeholders = ",".join([f"%({i})s" for i in range(len(chunk_ids))])
        query = f"""
            SELECT
                chunk_id,
                1 - (embedding <=> %(query_embedding)s::vector) AS similarity
            FROM chunk_embeddings
            WHERE chunk_id IN ({placeholders})
        """

        params: dict[str, str | int] = {"query_embedding": embedding_str}
        params.update({str(i): chunk_id for i, chunk_id in enumerate(chunk_ids)})  # type: ignore[arg-type]

        # Fetch similarities from database
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)  # type: ignore[arg-type]
                results: list = cur.fetchall()  # type: ignore[assignment]

        # Create similarity lookup
        similarity_map = {row["chunk_id"]: float(row["similarity"]) for row in results}

        # Update chunks with similarity scores and filter out chunks without embeddings
        reranked = []
        for chunk in chunks:
            if chunk.chunk_id in similarity_map:
                reranked_chunk = RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    score=similarity_map[chunk.chunk_id],
                    metadata=chunk.metadata,
                    ord=chunk.ord,
                    speaker=chunk.speaker,
                )
                reranked.append(reranked_chunk)

        # Sort by similarity descending
        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked

    def explain_query(
        self,
        query: str,
        filters: Optional[dict] = None,
        fts_candidates: int = 100,
        operator: Literal["and", "or"] = "or",
    ) -> str:
        """
        Get detailed explanation of hybrid retrieval process.

        Returns:
            Multi-line string explaining FTS + reranking stages
        """
        # Get FTS explain
        fts_explain = self.fts_retriever.explain_query(query, filters, operator)

        # Generate query embedding to show timing
        timer = Timer()
        timer.start()
        self.embedding_service.embed_text(query)
        embedding_ms = timer.stop()

        explanation = [
            "=" * 80,
            "HYBRID RETRIEVAL EXPLAIN",
            "=" * 80,
            "",
            "Stage 1: Full-Text Search (FTS)",
            f"  - Retrieves {fts_candidates} candidates using FTS with '{operator}' operator",
            "  - FTS EXPLAIN ANALYZE:",
            "",
            fts_explain,
            "",
            "Stage 2: Vector Reranking",
            f"  - Query embedding generation: {embedding_ms:.2f}ms",
            "  - Fetches embeddings for FTS candidates from chunk_embeddings table",
            "  - Computes cosine similarity between query and each candidate",
            "  - Reranks by similarity score (descending)",
            "  - Returns top N results",
            "",
            "=" * 80,
        ]

        return "\n".join(explanation)
