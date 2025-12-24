"""Vector similarity retrieval using pgvector and OpenAI embeddings."""

from typing import Optional

from src.database.connection import get_db_connection
from src.embeddings.service import EmbeddingService
from src.retrieval.models import RetrievalResponse, RetrievalResult
from src.utils.timing import Timer


class VectorSimilarityRetriever:
    """
    Vector similarity retrieval using pgvector cosine similarity.

    Uses OpenAI embeddings to convert queries to vectors and finds
    semantically similar chunks using cosine similarity.
    """

    def __init__(self):
        """Initialize with embedding service."""
        self.embedding_service = EmbeddingService()

    def retrieve(
        self,
        query: str,
        n: int = 50,
        filters: Optional[dict] = None,
    ) -> RetrievalResponse:
        """
        Retrieve top N chunks using vector similarity.

        Args:
            query: User query string
            n: Number of chunks to retrieve
            filters: Optional metadata filters (date ranges, doc_type, etc.)

        Returns:
            RetrievalResponse with chunks, timing, and query info
        """
        timer = Timer()

        # Step 1: Generate query embedding
        timer.start()
        query_embedding = self.embedding_service.embed_text(query)
        embedding_ms = timer.stop()

        # Step 2: Build SQL query with filters
        sql_query, params = self._build_query(query_embedding, n, filters)

        # Step 3: Execute retrieval
        timer.start()
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql_query, params)  # type: ignore[arg-type]
                results: list = cur.fetchall()  # type: ignore[assignment]
        retrieval_ms = timer.stop()

        # Step 4: Format results
        chunks = [
            RetrievalResult(
                chunk_id=row.get("chunk_id"),  # type: ignore[arg-type]
                doc_id=row.get("doc_id"),  # type: ignore[arg-type]
                text=row.get("text"),  # type: ignore[arg-type]
                score=float(row.get("similarity", 0)),  # type: ignore[arg-type]
                metadata={
                    "url": row.get("url"),  # type: ignore[arg-type]
                    "title": row.get("title"),  # type: ignore[arg-type]
                    "published_at": (
                        row.get("published_at").isoformat() if row.get("published_at") else None  # type: ignore[arg-type]
                    ),
                },
                ord=row.get("ord"),  # type: ignore[arg-type]
            )
            for row in results
        ]

        total_ms = embedding_ms + retrieval_ms

        return RetrievalResponse(
            chunks=chunks,
            timing_ms={
                "embedding": round(embedding_ms, 2),
                "retrieval": round(retrieval_ms, 2),
                "total": round(total_ms, 2),
            },
            query_info={
                "query": query,
                "n": n,
                "results_returned": len(chunks),
                "filters_applied": filters or {},
                "retrieval_mode": "vector",
            },
        )

    def _build_query(
        self, query_embedding: list[float], n: int, filters: Optional[dict]
    ) -> tuple[str, dict]:
        """
        Build SQL query for vector similarity search.

        Query structure:
        1. Join chunk_embeddings with chunks and docs
        2. Calculate cosine similarity using pgvector
        3. Apply metadata filters (WHERE clauses)
        4. Order by similarity (descending)
        5. Limit to top N

        Note: Cosine similarity ranges from -1 to 1, where 1 is most similar.
        """
        # Base query with vector similarity
        # pgvector's <=> operator computes cosine distance (1 - cosine_similarity)
        # We convert to similarity: 1 - distance = similarity
        sql = """
            SELECT
                c.id AS chunk_id,
                c.doc_id,
                c.text,
                c.ord,
                1 - (ce.embedding <=> %(query_embedding)s::vector) AS similarity,
                d.url,
                d.title,
                d.published_at,
                d.metadata
            FROM chunk_embeddings ce
            INNER JOIN chunks c ON ce.chunk_id = c.id
            INNER JOIN docs d ON c.doc_id = d.id
            WHERE TRUE
        """

        # Convert embedding list to pgvector format string '[1,2,3,...]'
        # pgvector expects this format for the ::vector cast
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        params = {"query_embedding": embedding_str, "n": n}

        # Apply metadata filters
        if filters:
            # Date range filter
            if filters.get("start_date"):
                sql += " AND d.published_at >= %(start_date)s"
                params["start_date"] = filters["start_date"]

            if filters.get("end_date"):
                sql += " AND d.published_at <= %(end_date)s"
                params["end_date"] = filters["end_date"]

            # Doc type filter
            if filters.get("doc_type"):
                sql += " AND d.doc_type = %(doc_type)s"
                params["doc_type"] = filters["doc_type"]

            # Source filter
            if filters.get("source"):
                sql += " AND d.source = %(source)s"
                params["source"] = filters["source"]

        # Order by similarity (highest first) and limit
        sql += """
            ORDER BY similarity DESC, c.id ASC
            LIMIT %(n)s
        """

        return sql, params

    def explain_query(
        self, query: str, filters: Optional[dict] = None
    ) -> tuple[str, float]:
        """
        Get EXPLAIN ANALYZE output for vector query performance analysis.

        Returns:
            Tuple of (explain_output, embedding_time_ms)
        """
        timer = Timer()

        # Generate query embedding
        timer.start()
        query_embedding = self.embedding_service.embed_text(query)
        embedding_ms = timer.stop()

        # Build query
        sql_query, params = self._build_query(query_embedding, n=50, filters=filters)
        explain_query_sql = f"EXPLAIN (ANALYZE, BUFFERS) {sql_query}"

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(explain_query_sql, params)  # type: ignore[arg-type]
                results: list = cur.fetchall()  # type: ignore[assignment]
                # EXPLAIN returns list of dicts with single key
                explain_output = "\n".join([str(list(row.values())[0]) if isinstance(row, dict) else str(row) for row in results])  # type: ignore[index]

        return explain_output, embedding_ms
