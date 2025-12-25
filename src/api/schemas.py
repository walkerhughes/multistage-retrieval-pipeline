from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, HttpUrl

from src.retrieval.models import RetrievalMode

# ============================================
# Ingestion Schemas
# ============================================


class IngestRequest(BaseModel):
    """Request body for POST /ingest

    Ingest a YouTube video by URL. The transcript will be automatically
    fetched, chunked into segments, and stored in the database.
    """

    url: HttpUrl = Field(
        ...,
        description="YouTube video URL (required)",
        examples=["https://www.youtube.com/watch?v=dQw4w9WgXcQ"]
    )


class TextIngestRequest(BaseModel):
    """Request body for POST /ingest/text

    Ingest raw text directly without fetching from YouTube. The text will be
    chunked into segments and stored in the database. Optionally include a
    title and metadata for better document organization and filtering.
    """

    text: str = Field(
        ...,
        description="Raw text to ingest (required, minimum 1 character)",
        min_length=1,
        examples=["Machine learning is a subset of artificial intelligence..."]
    )
    title: Optional[str] = Field(
        None,
        description="Optional title for the document (appears in metadata)",
        examples=["Introduction to Machine Learning"]
    )
    metadata: Optional[dict] = Field(
        default_factory=dict,
        description="Optional metadata as key-value pairs (e.g., author, source)",
        examples=[{"author": "John Doe", "source": "tutorial"}]
    )


class IngestResponse(BaseModel):
    """Response for POST /ingest and POST /ingest/text

    Returns metadata about the ingested document including processing statistics.
    """

    status: str = Field(
        "accepted",
        description="Status of the ingestion request"
    )
    doc_id: int = Field(
        ...,
        description="Unique database identifier for the ingested document"
    )
    url: str = Field(
        ...,
        description="URL of the ingested document (YouTube URL or internal reference)"
    )
    title: str = Field(
        ...,
        description="Title of the document"
    )
    chunk_count: int = Field(
        ...,
        description="Number of text chunks the document was split into"
    )
    total_tokens: int = Field(
        ...,
        description="Total number of tokens across all chunks (using GPT-4 tokenizer)"
    )
    ingestion_time_ms: float = Field(
        ...,
        description="Time taken to process the entire ingestion pipeline in milliseconds"
    )
    embeddings_generated: bool = Field(
        False,
        description="Whether semantic embeddings were generated for this document"
    )


# ============================================
# Query Schemas
# ============================================


class QueryFilters(BaseModel):
    """Optional metadata filters for queries

    All fields are optional. Use them to narrow down results to specific
    documents or time ranges.
    """

    source: Optional[str] = Field(
        None,
        description="Filter by document source (e.g., 'youtube', 'manual')",
        examples=["youtube"]
    )
    doc_type: Optional[str] = Field(
        None,
        description="Filter by document type (e.g., 'transcript', 'article')",
        examples=["transcript"]
    )
    start_date: Optional[datetime] = Field(
        None,
        description="Only return chunks from documents published on or after this date",
        examples=["2024-01-01T00:00:00"]
    )
    end_date: Optional[datetime] = Field(
        None,
        description="Only return chunks from documents published on or before this date",
        examples=["2024-12-31T23:59:59"]
    )


class QueryRequest(BaseModel):
    """Request body for POST /query

    Retrieve relevant text chunks from ingested documents using keyword search,
    semantic search, or a hybrid approach. Required parameter: 'query_text'.
    """

    q: str = Field(
        ...,
        description="Search query (required). Examples: 'machine learning basics', 'what is neural network'",
        min_length=1,
        examples=["what is machine learning?"]
    )
    n: int = Field(
        50,
        description="Maximum number of chunks to return (default: 50, range: 1-100)",
        ge=1,
        le=100,
        examples=[50]
    )
    mode: RetrievalMode = Field(
        RetrievalMode.FTS,
        description="Retrieval strategy: 'fts' (keyword search), 'vector' (semantic/embedding-based), "
                    "or 'hybrid' (keyword search + semantic reranking). Default: 'fts'",
        examples=["fts"]
    )
    operator: Literal["and", "or"] = Field(
        "or",
        description="Query operator for FTS mode (applies to 'fts' and 'hybrid' modes). "
                    "'or' = match any term (broad, default), 'and' = match all terms (strict). "
                    "E.g., 'machine learning': 'or' returns chunks with 'machine' OR 'learning', "
                    "'and' returns chunks with both words.",
        examples=["or"]
    )
    fts_candidates_for_reranking: int = Field(
        100,
        description="Number of initial keyword-matched chunks to rerank semantically. "
                    "Only used in 'hybrid' mode. Higher values = better recall but slower (range: 1-500). "
                    "Use 200+ for comprehensive recall, 50-100 for speed.",
        ge=1,
        le=500,
        examples=[100]
    )
    filters: Optional[QueryFilters] = Field(
        None,
        description="Optional metadata filters to narrow results by source, type, or date range"
    )


class ChunkResult(BaseModel):
    """A single text chunk returned from retrieval

    Represents one of the ranked text segments matching your query.
    """

    chunk_id: int = Field(
        ...,
        description="Unique identifier for this text chunk"
    )
    doc_id: int = Field(
        ...,
        description="ID of the parent document this chunk came from"
    )
    score: float = Field(
        ...,
        description="Relevance score (higher = more relevant to your query)"
    )
    text: str = Field(
        ...,
        description="The actual text content of this chunk"
    )
    metadata: dict = Field(
        ...,
        description="Document metadata (title, source, publication date, etc.)"
    )
    ord: int = Field(
        ...,
        description="Position of this chunk within the document (0-indexed)"
    )


class QueryResponse(BaseModel):
    """Response from POST /query endpoint

    Contains ranked chunks matching your query, along with timing and
    diagnostic information.
    """

    chunks: list[ChunkResult] = Field(
        ...,
        description="List of text chunks ranked by relevance to your query. "
                    "Order matters: first chunk is most relevant."
    )
    timing_ms: dict = Field(
        ...,
        description="Timing breakdown showing query execution time, retrieval time, "
                    "and reranking time (if applicable)"
    )
    query_info: dict = Field(
        ...,
        description="Metadata about how the query was executed (mode used, "
                    "number of candidates, filters applied, etc.)"
    )


# ============================================
# Benchmark Schemas
# ============================================


class BenchmarkResponse(BaseModel):
    """Response from GET /retrieval/bench endpoint

    Performance metrics and database execution plan for your query.
    """

    query_time_ms: float = Field(
        ...,
        description="Total time to execute the query in milliseconds "
                    "(includes embedding generation if using vector/hybrid modes)"
    )
    rows_returned: int = Field(
        ...,
        description="Number of matching chunks found by the query"
    )
    explain: str = Field(
        ...,
        description="PostgreSQL EXPLAIN ANALYZE output showing index usage, "
                    "scan types, and execution plan. Look for 'Bitmap Index Scan' "
                    "for optimal FTS performance."
    )
    query: str = Field(
        ...,
        description="Echo of the query string you provided"
    )
