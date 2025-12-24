import re
from typing import Literal, Optional

from src.database.connection import get_db_connection
from src.retrieval.models import RetrievalResponse, RetrievalResult
from src.utils.timing import Timer

# Common English stop words that Postgres FTS removes
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "this", "to", "was", "were", "will", "with", "not", "but",
    "they", "have", "been", "would", "could", "should", "their", "there",
}


class FullTextSearchRetriever:
    """
    Postgres FTS-based retrieval using tsvector and websearch_to_tsquery.
    """

    def retrieve(
        self,
        query: str,
        n: int = 50,
        filters: Optional[dict] = None,
        operator: Literal["and", "or"] = "or",
    ) -> RetrievalResponse:
        """
        Retrieve top N chunks using Full-Text Search.

        Args:
            query: User query string
            n: Number of chunks to retrieve
            filters: Optional metadata filters (date ranges, doc_type, etc.)
            operator: "or" for broad retrieval (any term matches),
                      "and" for strict retrieval (all terms must match).
                      Defaults to "or" for broad recall.

        Returns:
            RetrievalResponse with chunks, timing, and query info
        """
        timer = Timer()

        # Build SQL query with filters
        sql_query, params = self._build_query(query, n, filters, operator)

        # Execute retrieval
        timer.start()
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql_query, params)  # type: ignore[arg-type]
                results: list = cur.fetchall()
        retrieval_ms = timer.stop()

        # Format results
        chunks = [
            RetrievalResult(
                chunk_id=row.get("chunk_id"),  # type: ignore[arg-type]
                doc_id=row.get("doc_id"),  # type: ignore[arg-type]
                text=row.get("text"),  # type: ignore[arg-type]
                score=float(row.get("score", 0)),  # type: ignore[arg-type]
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

        return RetrievalResponse(
            chunks=chunks,
            timing_ms={
                "retrieval": round(retrieval_ms, 2),
                "total": round(retrieval_ms, 2),
            },
            query_info={
                "query": query,
                "n": n,
                "results_returned": len(chunks),
                "filters_applied": filters or {},
            },
        )

    def _build_or_tsquery(self, query: str) -> Optional[str]:
        """
        Build an OR-based tsquery string from a natural language query.

        Extracts words, removes stop words, and joins with ' | ' for OR logic.
        Returns a format suitable for to_tsquery().

        Example: "reinforcement learning research" -> "reinforcement | learning | research"
        """
        # Extract words (alphanumeric only)
        words = re.findall(r'\b[a-zA-Z0-9]+\b', query.lower())
        # Filter out stop words and very short words
        meaningful_words = [w for w in words if w not in STOP_WORDS and len(w) > 1]

        if not meaningful_words:
            # Fallback: use original query with websearch
            return None

        # Join with OR operator for to_tsquery
        return " | ".join(meaningful_words)

    def _build_query(
        self, query: str, n: int, filters: Optional[dict], operator: str = "or"
    ) -> tuple[str, dict]:
        """
        Build SQL query with FTS and metadata filters.

        Query structure:
        1. Join chunks with docs for metadata
        2. Apply FTS using to_tsquery (OR) or websearch_to_tsquery (AND)
        3. Apply metadata filters (WHERE clauses)
        4. Rank by ts_rank
        5. Limit to top N
        """
        params = {"query": query, "n": n}

        # Choose tsquery function based on operator
        if operator == "or":
            or_query = self._build_or_tsquery(query)
            if or_query:
                params["or_query"] = or_query
                tsquery_expr = "to_tsquery('english', %(or_query)s)"
            else:
                # Fallback to websearch if no meaningful terms
                tsquery_expr = "websearch_to_tsquery('english', %(query)s)"
        else:
            # AND logic uses websearch_to_tsquery
            tsquery_expr = "websearch_to_tsquery('english', %(query)s)"

        # Base query with FTS
        sql = f"""
            SELECT
                c.id AS chunk_id,
                c.doc_id,
                c.text,
                c.ord,
                ts_rank(c.tsv, {tsquery_expr}) AS score,
                d.url,
                d.title,
                d.published_at,
                d.metadata
            FROM chunks c
            INNER JOIN docs d ON c.doc_id = d.id
            WHERE c.tsv @@ {tsquery_expr}
        """

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

        # Order by rank and limit
        sql += """
            ORDER BY score DESC, c.id ASC
            LIMIT %(n)s
        """

        return sql, params

    def explain_query(
        self,
        query: str,
        filters: Optional[dict] = None,
        operator: Literal["and", "or"] = "or",
    ) -> str:
        """
        Get EXPLAIN ANALYZE output for query performance analysis.
        """
        sql_query, params = self._build_query(query, n=50, filters=filters, operator=operator)
        explain_query_sql = f"EXPLAIN (ANALYZE, BUFFERS) {sql_query}"

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(explain_query_sql, params)  # type: ignore[arg-type]
                results: list = cur.fetchall()
                # EXPLAIN returns list of dicts with single key
                return "\n".join([str(list(row.values())[0]) if isinstance(row, dict) else str(row) for row in results])  # type: ignore[index]
