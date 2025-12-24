from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, HttpUrl

from src.retrieval.models import RetrievalMode

# ============================================
# Ingestion Schemas
# ============================================


class IngestRequest(BaseModel):
    """Request body for POST /ingest"""

    url: HttpUrl = Field(..., description="YouTube video URL")


class TextIngestRequest(BaseModel):
    """Request body for POST /ingest/text"""

    text: str = Field(..., description="Raw text to ingest", min_length=1)
    title: Optional[str] = Field(None, description="Optional title for the document")
    metadata: Optional[dict] = Field(
        default_factory=dict, description="Optional metadata"
    )


class IngestResponse(BaseModel):
    """Response for POST /ingest"""

    status: str = "accepted"
    doc_id: int
    url: str
    title: str
    chunk_count: int
    total_tokens: int
    ingestion_time_ms: float
    embeddings_generated: bool = False


# ============================================
# Query Schemas
# ============================================


class QueryFilters(BaseModel):
    """Optional metadata filters for queries"""

    source: Optional[str] = None
    doc_type: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


class QueryRequest(BaseModel):
    """Request body for POST /query"""

    q: str = Field(..., description="User query string", min_length=1)
    n: int = Field(50, description="Number of chunks to retrieve", ge=1, le=100)
    mode: RetrievalMode = Field(
        RetrievalMode.FTS,
        description="Retrieval mode: fts, vector, or hybrid",
    )
    operator: Literal["and", "or"] = Field(
        "or",
        description="FTS term logic: 'or' for broad retrieval (any term matches), "
        "'and' for strict (all terms must match). Applies to FTS and Hybrid modes.",
    )
    fts_candidates: int = Field(
        100,
        description="Number of FTS candidates to retrieve before reranking. "
        "Only applies to Hybrid mode. Higher values increase recall but cost more.",
        ge=1,
        le=500,
    )
    filters: Optional[QueryFilters] = None


class ChunkResult(BaseModel):
    """Single chunk result"""

    chunk_id: int
    doc_id: int
    score: float
    text: str
    metadata: dict
    ord: int


class QueryResponse(BaseModel):
    """Response for POST /query"""

    chunks: list[ChunkResult]
    timing_ms: dict
    query_info: dict


# ============================================
# Benchmark Schemas
# ============================================


class BenchmarkResponse(BaseModel):
    """Response for GET /bench/retrieval"""

    query_time_ms: float
    rows_returned: int
    explain: str
    query: str
