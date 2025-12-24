from fastapi import APIRouter, HTTPException

from src.api.schemas import IngestRequest, IngestResponse, TextIngestRequest
from src.ingestion.pipeline import IngestionPipeline

router = APIRouter(prefix="/ingest", tags=["Ingestion"])


@router.post("", response_model=IngestResponse)
async def ingest_youtube(request: IngestRequest):
    """
    Ingest a YouTube video transcript.

    Steps:
    1. Fetch transcript from YouTube
    2. Chunk into token-based segments
    3. Store in Postgres (docs + chunks tables)

    Returns ingestion metadata and timing.
    """
    try:
        pipeline = IngestionPipeline()
        result = pipeline.ingest_youtube_url(str(request.url))
        return IngestResponse(**result)

    except ValueError as e:
        # Invalid URL or transcript unavailable
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        # Unexpected errors
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.post("/text", response_model=IngestResponse)
async def ingest_text(request: TextIngestRequest):
    """
    Ingest raw text directly.

    Steps:
    1. Create document entry
    2. Chunk text into token-based segments
    3. Generate embeddings for each chunk
    4. Store in Postgres (docs + chunks + chunk_embeddings tables)

    Returns ingestion metadata and timing.
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
