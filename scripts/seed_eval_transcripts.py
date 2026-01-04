#!/usr/bin/env python
"""Seed database with eval transcripts for CI testing.

Loads transcripts from evals/datasets/transcripts/ and ingests them
into the database using the ingestion pipeline.

Usage:
    python scripts/seed_eval_transcripts.py
"""

from evals.loaders import TranscriptLoader
from src.database.connection import close_db_pool, init_db_pool
from src.ingestion.pipeline import IngestionPipeline


def seed_transcripts() -> None:
    """Load and ingest all eval transcripts."""
    # Initialize database
    init_db_pool()

    try:
        # Load transcripts using default path (evals/datasets/transcripts/)
        loader = TranscriptLoader()
        transcripts = loader.load_all()

        print(f"Found {len(transcripts)} transcripts to ingest")

        # Ingest each transcript
        pipeline = IngestionPipeline(generate_embeddings=True)

        for transcript in transcripts:
            print(f"\nIngesting: {transcript.filename}")
            result = pipeline.ingest_raw_text(
                text=transcript.text,
                title=transcript.filename.replace(".md", ""),
                metadata={
                    "speaker": transcript.speaker,
                    "source": "eval_transcript",
                },
            )
            print(f"  -> doc_id={result['doc_id']}, chunks={result['chunk_count']}")

        print("\nSeeding complete!")

    finally:
        close_db_pool()


if __name__ == "__main__":
    seed_transcripts()
