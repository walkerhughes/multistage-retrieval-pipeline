from fastapi import APIRouter, HTTPException

from src.api.schemas import IngestRequest, IngestResponse, TextIngestRequest
from src.ingestion.pipeline import IngestionPipeline

router = APIRouter(prefix="/ingest", tags=["Ingestion"])


@router.post("/youtube", response_model=IngestResponse)
async def ingest_youtube(request: IngestRequest):
    """Ingest a YouTube video transcript into the database

    This endpoint fetches the transcript from a YouTube video URL, cleans the text,
    breaks it into manageable text chunks, and stores them in the database for later
    retrieval.

    **What happens:**
    1. Validates the YouTube URL
    2. Fetches the video transcript (auto-generated or manual captions)
    3. Cleans transcript text (removes newlines and backslashes)
    4. Chunks transcript into segments (400-800 tokens, using GPT-4 tokenizer)
    5. Stores in Postgres for fast full-text and semantic search

    **Required:** YouTube URL must be a valid, publicly accessible video.

    **Optional:**
    - title: Override the YouTube video title
    - metadata: Custom metadata to merge with YouTube metadata (e.g., category, tags)

    **Note:** YouTube may block transcript requests. Use POST /ingest/text
    as a fallback to ingest raw text directly.

    **Returns:** Document metadata including ID, chunk count, token count,
    and processing time in milliseconds.
    """
    try:
        pipeline = IngestionPipeline()
        result = pipeline.ingest_youtube_url(
            url=str(request.url), title=request.title, metadata=request.metadata
        )
        return IngestResponse(**result)

    except ValueError as e:
        # Invalid URL or transcript unavailable
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        # Unexpected errors
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.post("/text", response_model=IngestResponse)
async def ingest_text(request: TextIngestRequest):
    """Ingest raw text directly into the database

    This endpoint accepts raw text content and stores it in the database for
    retrieval. Use this when you want to ingest text without fetching from
    YouTube (e.g., articles, documentation, custom content).

    **What happens:**
    1. Creates a document entry with optional title and metadata
    2. Chunks text into segments (400-800 tokens, using GPT-4 tokenizer)
    3. Generates semantic embeddings for each chunk (if enabled)
    4. Stores in Postgres for fast full-text and semantic search

    **Required:** Raw text content (minimum 1 character)

    **Optional:**
    - title: Document title (for organization and filtering)
    - metadata: Custom key-value pairs (author, source, category, etc.)

    **Returns:** Document metadata including ID, chunk count, token count,
    embedding generation status, and processing time in milliseconds.
    """
    try:
        pipeline = IngestionPipeline()
        result = pipeline.ingest_raw_text(
            text=request.text, title=request.title, metadata=request.metadata
        )
        return IngestResponse(**result)

    except ValueError as e:
        # Invalid input
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        # Unexpected errors
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
