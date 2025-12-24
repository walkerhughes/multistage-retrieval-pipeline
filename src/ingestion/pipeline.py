import json
from datetime import datetime

from src.database.connection import execute_insert, get_db_connection
from src.embeddings.service import EmbeddingService
from src.ingestion.chunker import TokenBasedChunker
from src.ingestion.youtube_loader import YouTubeTranscriptFetcher


class IngestionPipeline:
    """End-to-end ingestion pipeline: fetch → chunk → embed → store."""

    def __init__(self, generate_embeddings: bool = True):
        self.fetcher = YouTubeTranscriptFetcher()
        self.chunker = TokenBasedChunker()
        self.generate_embeddings = generate_embeddings
        self.embedding_service: EmbeddingService | None = None

        if generate_embeddings:
            try:
                self.embedding_service = EmbeddingService()
            except ValueError as e:
                print(f"Warning: {e}")
                print("Continuing without embeddings...")
                self.generate_embeddings = False

    def ingest_youtube_url(self, url: str) -> dict:
        """
        Ingest a YouTube video transcript.

        Steps:
        1. Fetch transcript and metadata
        2. Insert into docs table
        3. Chunk transcript
        4. Insert chunks into chunks table

        Args:
            url: YouTube video URL

        Returns:
            Dict with doc_id, chunk_count, and timing info
        """
        start_time = datetime.now()

        # Step 1: Fetch transcript
        doc = self.fetcher.fetch(url)

        # Step 2: Insert document
        doc_id = self._insert_document(doc)

        # Step 3: Chunk text
        chunks = self.chunker.chunk(doc.text)

        # Step 4: Insert chunks
        chunk_ids = self._insert_chunks(doc_id, chunks)

        # Step 5: Generate and insert embeddings (if enabled)
        embeddings_generated = False
        if self.generate_embeddings and self.embedding_service:
            print("\nGenerating embeddings...")
            self._generate_and_insert_embeddings(chunk_ids, chunks)
            embeddings_generated = True

        end_time = datetime.now()
        elapsed_ms = (end_time - start_time).total_seconds() * 1000

        return {
            "doc_id": doc_id,
            "url": url,
            "title": doc.title,
            "chunk_count": len(chunks),
            "total_tokens": sum(c.token_count for c in chunks),
            "ingestion_time_ms": round(elapsed_ms, 2),
            "embeddings_generated": embeddings_generated,
        }

    def ingest_raw_text(
        self, text: str, title: str | None = None, metadata: dict | None = None
    ) -> dict:
        """
        Ingest raw text directly (not from YouTube).

        Steps:
        1. Create document entry
        2. Chunk text
        3. Insert chunks
        4. Generate and insert embeddings

        Args:
            text: Raw text to ingest
            title: Optional title for the document
            metadata: Optional metadata dict

        Returns:
            Dict with doc_id, chunk_count, and timing info
        """
        start_time = datetime.now()

        # Step 1: Insert document
        doc_id = self._insert_raw_document(text, title, metadata)

        # Step 2: Chunk text
        chunks = self.chunker.chunk(text)

        # Step 3: Insert chunks
        chunk_ids = self._insert_chunks(doc_id, chunks)

        # Step 4: Generate and insert embeddings (if enabled)
        embeddings_generated = False
        if self.generate_embeddings and self.embedding_service:
            print("\nGenerating embeddings...")
            self._generate_and_insert_embeddings(chunk_ids, chunks)
            embeddings_generated = True

        end_time = datetime.now()
        elapsed_ms = (end_time - start_time).total_seconds() * 1000

        return {
            "doc_id": doc_id,
            "url": "n/a",
            "title": title or "Untitled Document",
            "chunk_count": len(chunks),
            "total_tokens": sum(c.token_count for c in chunks),
            "ingestion_time_ms": round(elapsed_ms, 2),
            "embeddings_generated": embeddings_generated,
        }

    def _insert_document(self, doc) -> int:
        """Insert document into docs table."""
        query = """
            INSERT INTO docs (source, url, title, doc_type, published_at, raw_text, metadata)
            VALUES (%(source)s, %(url)s, %(title)s, %(doc_type)s, %(published_at)s, %(raw_text)s, %(metadata)s)
            RETURNING id
        """

        params = {
            "source": "youtube",
            "url": doc.url,
            "title": doc.title,
            "doc_type": "transcript",
            "published_at": doc.published_at,
            "raw_text": doc.text,
            "metadata": json.dumps(doc.metadata),
        }

        return execute_insert(query, params)

    def _insert_raw_document(
        self, text: str, title: str | None, metadata: dict | None
    ) -> int:
        """Insert raw text document into docs table."""
        query = """
            INSERT INTO docs (source, url, title, doc_type, published_at, raw_text, metadata)
            VALUES (%(source)s, %(url)s, %(title)s, %(doc_type)s, %(published_at)s, %(raw_text)s, %(metadata)s)
            RETURNING id
        """

        params = {
            "source": "text",
            "url": "n/a",
            "title": title or "Untitled Document",
            "doc_type": "text",
            "published_at": datetime.now(),
            "raw_text": text,
            "metadata": json.dumps(metadata or {}),
        }

        return execute_insert(query, params)

    def _insert_chunks(self, doc_id: int, chunks: list) -> list[int]:
        """Batch insert chunks into chunks table and return chunk IDs."""
        if not chunks:
            return []

        query = """
            INSERT INTO chunks (doc_id, ord, text, token_count)
            VALUES (%(doc_id)s, %(ord)s, %(text)s, %(token_count)s)
            RETURNING id
        """

        chunk_ids = []
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Batch insert for efficiency
                for chunk in chunks:
                    cur.execute(
                        query,  # type: ignore[arg-type]
                        {
                            "doc_id": doc_id,
                            "ord": chunk.ord,
                            "text": chunk.text,
                            "token_count": chunk.token_count,
                        },
                    )
                    result: dict | None = cur.fetchone()  # type: ignore[assignment]
                    if result:
                        chunk_ids.append(result.get("id"))  # type: ignore[arg-type]
                conn.commit()

        return chunk_ids

    def _generate_and_insert_embeddings(
        self, chunk_ids: list[int], chunks: list
    ) -> None:
        """Generate embeddings and insert into chunk_embeddings table."""
        if not chunk_ids or not chunks or not self.embedding_service:
            return

        # Extract texts from chunks
        texts = [chunk.text for chunk in chunks]

        # Generate embeddings in batch
        embeddings = self.embedding_service.embed_batch(texts)

        # Insert embeddings
        query = """
            INSERT INTO chunk_embeddings (chunk_id, embedding)
            VALUES (%(chunk_id)s, %(embedding)s)
        """

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                for chunk_id, embedding in zip(chunk_ids, embeddings):
                    cur.execute(query, {"chunk_id": chunk_id, "embedding": embedding})
                conn.commit()

        print(f"✓ Generated and inserted {len(embeddings)} embeddings")
