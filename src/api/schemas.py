from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

from src.agents.models import AgentType
from src.retrieval.models import RetrievalMode

# ============================================
# Ingestion Schemas
# ============================================


class TextIngestRequest(BaseModel):
    """Request body for POST /ingest/text

    Ingest raw text directly. The text will be chunked into segments and
    stored in the database. Optionally include a title and metadata for
    better document organization and filtering.
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
    """Response for POST /ingest/text

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
        description="URL of the ingested document (source URL or internal reference)"
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

    query: str = Field(
        ...,
        description="Search query (required). Examples: 'machine learning basics', 'what is neural network'",
        min_length=1,
        examples=["what is machine learning?"]
    )
    max_returned: int = Field(
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
    fts_candidates: int = Field(
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


# ============================================
# Chat Completion Schemas
# ============================================


class ChatCompletionRequest(BaseModel):
    """Request body for POST /chat/completion

    Generate an answer to a question using the RAG pipeline. Choose an agent
    type and configure retrieval parameters for evaluation purposes.
    """

    question: str = Field(
        ...,
        description="User question to answer (required)",
        min_length=1,
        max_length=2000,
        examples=["What have Dwarkesh's guests said about the timeline for AGI?"]
    )
    agent: AgentType = Field(
        AgentType.VANILLA,
        description="Agent type to use: 'vanilla' (single-query RAG) or "
                    "'multi-query' (query decomposition, not yet implemented). Default: 'vanilla'",
        examples=["vanilla"]
    )
    mode: RetrievalMode = Field(
        RetrievalMode.HYBRID,
        description="Retrieval mode: 'fts' (keyword), 'vector' (semantic), or 'hybrid' (combined). "
                    "Default: 'hybrid'",
        examples=["hybrid"]
    )
    operator: Literal["and", "or"] = Field(
        "or",
        description="FTS query operator: 'or' (match any term) or 'and' (match all terms). "
                    "Default: 'or'",
        examples=["or"]
    )
    fts_candidates: int = Field(
        100,
        description="Number of FTS candidates for hybrid reranking (range: 1-500). Default: 100",
        ge=1,
        le=500,
        examples=[100]
    )
    max_returned: int = Field(
        10,
        description="Maximum chunks to retrieve for context (range: 1-100). Default: 10",
        ge=1,
        le=100,
        examples=[10]
    )
    filters: Optional[QueryFilters] = Field(
        None,
        description="Optional metadata filters (source, doc_type, date range)"
    )


class RetrievedChunkResponse(BaseModel):
    """A chunk used as context for answer generation.

    Represents a text segment retrieved from the database and used
    to ground the generated answer.
    """

    chunk_id: int = Field(
        ...,
        description="Unique identifier for this chunk"
    )
    doc_id: int = Field(
        ...,
        description="ID of the parent document"
    )
    text: str = Field(
        ...,
        description="Text content of the chunk"
    )
    score: float = Field(
        ...,
        description="Relevance score from retrieval"
    )
    metadata: dict = Field(
        ...,
        description="Document metadata (title, source, etc.)"
    )
    ord: int = Field(
        ...,
        description="Position within the parent document"
    )


class ChatCompletionResponse(BaseModel):
    """Response from POST /chat/completion

    Contains the generated answer along with metadata about the RAG pipeline
    execution including retrieved chunks, timing, and token usage.

    Multi-query specific fields (sub_queries, chunks_per_subquery, deduplication_stats)
    are only populated when using the 'multi-query' agent.
    """

    answer: str = Field(
        ...,
        description="Generated answer to the question"
    )
    trace_id: Optional[str] = Field(
        None,
        description="LangSmith trace ID for debugging and observability. "
                    "Use to inspect the full execution trace in LangSmith dashboard."
    )
    latency_ms: float = Field(
        ...,
        description="Total agent execution time in milliseconds (retrieval + generation)"
    )
    retrieved_chunks: list[RetrievedChunkResponse] = Field(
        ...,
        description="Chunks used as context for answer generation. "
                    "Inspect these to verify answer grounding."
    )
    model_used: str = Field(
        ...,
        description="LLM model used for answer generation"
    )
    tokens_used: dict = Field(
        ...,
        description="Token usage breakdown: prompt_tokens, completion_tokens, total_tokens"
    )
    # Multi-query specific fields (only populated for agent='multi-query')
    sub_queries: list[str] = Field(
        default_factory=list,
        description="Sub-queries generated by the multi-query agent. "
                    "Empty for vanilla agent."
    )
    chunks_per_subquery: dict[str, int] = Field(
        default_factory=dict,
        description="Number of chunks retrieved for each sub-query. "
                    "Empty for vanilla agent."
    )
    deduplication_stats: dict = Field(
        default_factory=dict,
        description="Statistics about chunk deduplication (total_before_dedup, unique_chunks, "
                    "duplicates_removed, chunks_boosted, etc.). Empty for vanilla agent."
    )


# ============================================
# Chunk Expansion Schemas
# ============================================


class ExpandRequest(BaseModel):
    """Request body for POST /retrieval/expand

    Expand chunk IDs to retrieve full speaker turn context.
    """

    chunk_ids: list[int] = Field(
        ...,
        description="List of chunk IDs to expand (required, minimum 1 ID)",
        min_length=1,
        examples=[[123, 456, 789]]
    )


class TurnData(BaseModel):
    """Speaker turn data with full context."""

    turn_id: int = Field(..., description="Unique turn identifier")
    doc_id: int = Field(..., description="Parent document ID")
    ord: int = Field(..., description="Turn order within document (0-indexed)")
    speaker: str = Field(..., description="Speaker name")
    full_text: str = Field(..., description="Complete turn text (not chunked)")
    start_time_seconds: int | None = Field(None, description="Turn start time in seconds")
    section_title: str | None = Field(None, description="Section/topic title")
    token_count: int = Field(..., description="Total tokens in turn")


class ExpandResponse(BaseModel):
    """Response from POST /retrieval/expand endpoint."""

    turns: list[TurnData] = Field(
        ...,
        description="List of turn data for requested chunks. "
                    "Deduplicated - same turn appears once even if multiple chunks reference it."
    )
    total_turns: int = Field(..., description="Number of unique turns returned")
    query_time_ms: float = Field(
        ...,
        description="Time taken to expand chunks and retrieve turn data in milliseconds"
    )


class QAPairsRequest(BaseModel):
    """Request body for POST /retrieval/qa-pairs

    Generate Q&A pairs from turn IDs by pairing with previous turn.
    """

    turn_ids: list[int] = Field(
        ...,
        description="List of turn IDs to generate Q&A pairs from (required, minimum 1 ID)",
        min_length=1,
        examples=[[123, 456, 789]]
    )


class QAPair(BaseModel):
    """Question-Answer pair from consecutive speaker turns."""

    question_turn: TurnData = Field(..., description="Previous turn (question context)")
    answer_turn: TurnData = Field(..., description="Target turn (answer)")


class QAPairsResponse(BaseModel):
    """Response from POST /retrieval/qa-pairs endpoint."""

    pairs: list[QAPair] = Field(
        ...,
        description="List of Q&A pairs. Pairs are only created when a previous turn exists "
                    "within the same document."
    )
    total_pairs: int = Field(..., description="Number of Q&A pairs returned")
    skipped_turns: int = Field(
        ...,
        description="Number of valid turns skipped (first turn in doc, no previous turn available)"
    )
    not_found_count: int = Field(
        ...,
        description="Number of requested turn IDs not found in the database"
    )
    query_time_ms: float = Field(
        ...,
        description="Time taken to generate Q&A pairs in milliseconds"
    )
