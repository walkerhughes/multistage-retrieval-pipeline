#!/usr/bin/env python
"""Seed database with eval transcripts for CI testing.

Loads transcripts from evals/tasks/retrieval/datasets/transcripts/ and ingests them
into the database using the ingestion pipeline.

Usage:
    python -m evals.tasks.retrieval.seed_transcripts
"""

import sys

from evals.tasks.retrieval.loaders import TranscriptLoader
from src.database.connection import close_db_pool, get_db_connection, init_db_pool
from src.ingestion.pipeline import IngestionPipeline


def seed_transcripts() -> None:
    """Load and ingest all eval transcripts."""
    print("=" * 60)
    print("SEEDING EVAL TRANSCRIPTS")
    print("=" * 60)

    # Initialize database
    print("\nInitializing database connection...")
    init_db_pool()

    try:
        # Clear existing data to ensure consistent chunk IDs
        print("\nClearing existing data...")
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Delete in order to respect foreign keys
                cur.execute("DELETE FROM chunk_embeddings")
                cur.execute("DELETE FROM chunks")
                cur.execute("DELETE FROM docs")
                # Reset sequences so IDs start at 1
                cur.execute("ALTER SEQUENCE docs_id_seq RESTART WITH 1")
                cur.execute("ALTER SEQUENCE chunks_id_seq RESTART WITH 1")
                conn.commit()
        print("Database cleared, sequences reset to 1")

        # Load transcripts using default path (evals/datasets/transcripts/)
        loader = TranscriptLoader()
        print(f"\nLoading transcripts from: {loader.transcripts_dir}")
        transcripts = loader.load_all()

        if not transcripts:
            print("ERROR: No transcripts found!")
            sys.exit(1)

        print(f"Found {len(transcripts)} transcripts to ingest:")
        for t in transcripts:
            print(f"  - {t.filename}")

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
            print(f"  -> doc_id={result['doc_id']}, chunks={result['chunk_count']}, embeddings={result['embeddings_generated']}")

        # Verify final state
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) as count FROM docs")
                docs = cur.fetchone()
                cur.execute("SELECT COUNT(*) as count FROM chunks")
                chunks = cur.fetchone()
                cur.execute("SELECT COUNT(*) as count FROM chunk_embeddings")
                embeddings = cur.fetchone()

        print("\n" + "=" * 60)
        print("SEEDING COMPLETE")
        print("=" * 60)
        print(f"Total docs: {docs['count']}")
        print(f"Total chunks: {chunks['count']}")
        print(f"Total embeddings: {embeddings['count']}")

    except Exception as e:
        print(f"\nERROR during seeding: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        close_db_pool()


if __name__ == "__main__":
    seed_transcripts()
