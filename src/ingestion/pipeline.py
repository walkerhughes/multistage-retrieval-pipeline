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

    def ingest_youtube_url(
        self, url: str, title: str | None = None, metadata: dict | None = None
    ) -> dict:
        """
        Ingest a YouTube video transcript.

        Steps:
        1. Fetch transcript and metadata
        2. Insert into docs table
        3. Chunk transcript
        4. Insert chunks into chunks table

        Args:
            url: YouTube video URL
            title: Optional title override (uses YouTube video title if not provided)
            metadata: Optional custom metadata to merge with YouTube metadata

        Returns:
            Dict with doc_id, chunk_count, and timing info
        """
        start_time = datetime.now()

        # Step 1: Fetch transcript
        doc = self.fetcher.fetch(url)

        # Step 2: Insert document (with optional overrides)
        doc_id = self._insert_document(doc, title_override=title, custom_metadata=metadata)
        if doc_id is None:
            raise ValueError("Failed to insert document into database")

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

        # Use override title if provided, otherwise use YouTube title
        final_title = title if title else doc.title

        return {
            "doc_id": doc_id,
            "url": url,
            "title": final_title,
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
        if doc_id is None:
            raise ValueError("Failed to insert document into database")

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

    def _insert_document(
        self, doc, title_override: str | None = None, custom_metadata: dict | None = None
    ) -> int | None:
        """
        Insert document into docs table.

        Args:
            doc: YouTubeDocument instance
            title_override: Optional title to use instead of doc.title
            custom_metadata: Optional custom metadata to merge with doc.metadata

        Returns:
            Document ID (or None if insert fails)
        """
        query = """
            INSERT INTO docs (source, url, title, doc_type, published_at, raw_text, metadata)
            VALUES (%(source)s, %(url)s, %(title)s, %(doc_type)s, %(published_at)s, %(raw_text)s, %(metadata)s)
            RETURNING id
        """

        # Use override title if provided, otherwise use document title
        final_title = title_override if title_override else doc.title

        # Merge custom metadata with YouTube metadata
        final_metadata = {**doc.metadata}
        if custom_metadata:
            final_metadata.update(custom_metadata)

        params = {
            "source": "youtube",
            "url": doc.url,
            "title": final_title,
            "doc_type": "transcript",
            "published_at": doc.published_at,
            "raw_text": doc.text,
            "metadata": json.dumps(final_metadata),
        }

        return execute_insert(query, params)

    def _insert_raw_document(
        self, text: str, title: str | None, metadata: dict | None
    ) -> int | None:
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

    def ingest_with_turns(
        self,
        turns: list[dict],
        title: str,
        url: str,
        published_at: datetime | None = None,
        metadata: dict | None = None,
        doc_type: str = "transcript",
    ) -> dict:
        """
        Ingest transcript with speaker turn structure.

        This method preserves speaker attribution by:
        1. Inserting full speaker turns into the turns table
        2. Chunking each turn individually (one turn = one or more chunks)
        3. Linking chunks back to their source turn via turn_id

        Args:
            turns: List of turn dicts with keys:
                - speaker: str
                - start_time_seconds: int | None
                - text: str
                - section_title: str | None
                - ord: int
            title: Document title
            url: Source URL
            published_at: Optional publish datetime
            metadata: Optional metadata dict
            doc_type: Type of document ('transcript' or 'blog')

        Returns:
            Dict with doc_id, turn_count, chunk_count, and timing info
        """
        start_time = datetime.now()

        # Step 1: Build raw_text from turns
        raw_text = "\n\n".join(t["text"] for t in turns)

        # Step 2: Insert document
        doc_id = self._insert_podcast_document(
            url=url,
            title=title,
            raw_text=raw_text,
            published_at=published_at,
            metadata=metadata,
            doc_type=doc_type,
        )
        if doc_id is None:
            raise ValueError("Failed to insert document into database")

        # Step 3: Insert turns
        turn_ids = self._insert_turns(doc_id, turns)

        # Step 4: Chunk each turn and insert with turn_id
        all_chunk_ids = []
        all_chunks = []
        global_ord = 0

        for turn_data, turn_id in zip(turns, turn_ids):
            # Chunk the turn text
            turn_chunks = self.chunker.chunk(turn_data["text"])

            # Insert chunks with turn_id
            for chunk in turn_chunks:
                chunk_id = self._insert_chunk_with_turn(
                    doc_id=doc_id,
                    turn_id=turn_id,
                    ord=global_ord,
                    text=chunk.text,
                    token_count=chunk.token_count,
                )
                if chunk_id:
                    all_chunk_ids.append(chunk_id)
                    all_chunks.append(chunk)
                global_ord += 1

        # Step 5: Generate and insert embeddings (if enabled)
        embeddings_generated = False
        if self.generate_embeddings and self.embedding_service and all_chunks:
            print("\nGenerating embeddings...")
            self._generate_and_insert_embeddings(all_chunk_ids, all_chunks)
            embeddings_generated = True

        end_time = datetime.now()
        elapsed_ms = (end_time - start_time).total_seconds() * 1000

        return {
            "doc_id": doc_id,
            "url": url,
            "title": title,
            "turn_count": len(turns),
            "chunk_count": len(all_chunks),
            "total_tokens": sum(c.token_count for c in all_chunks),
            "ingestion_time_ms": round(elapsed_ms, 2),
            "embeddings_generated": embeddings_generated,
        }

    def _insert_podcast_document(
        self,
        url: str,
        title: str,
        raw_text: str,
        published_at: datetime | None = None,
        metadata: dict | None = None,
        doc_type: str = "transcript",
    ) -> int | None:
        """Insert podcast document into docs table."""
        query = """
            INSERT INTO docs (source, url, title, doc_type, published_at, raw_text, metadata)
            VALUES (%(source)s, %(url)s, %(title)s, %(doc_type)s, %(published_at)s, %(raw_text)s, %(metadata)s)
            RETURNING id
        """

        params = {
            "source": "dwarkesh",
            "url": url,
            "title": title,
            "doc_type": doc_type,
            "published_at": published_at or datetime.now(),
            "raw_text": raw_text,
            "metadata": json.dumps(metadata or {}),
        }

        return execute_insert(query, params)

    def _insert_turns(self, doc_id: int, turns: list[dict]) -> list[int]:
        """Batch insert turns into turns table and return turn IDs."""
        if not turns:
            return []

        query = """
            INSERT INTO turns (doc_id, ord, speaker, start_time_seconds, text, section_title, token_count, metadata)
            VALUES (%(doc_id)s, %(ord)s, %(speaker)s, %(start_time_seconds)s, %(text)s, %(section_title)s, %(token_count)s, %(metadata)s)
            RETURNING id
        """

        turn_ids = []
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                for i, turn in enumerate(turns):
                    # Count tokens for the turn
                    token_count = self.chunker.count_tokens(turn["text"])

                    cur.execute(
                        query,  # type: ignore[arg-type]
                        {
                            "doc_id": doc_id,
                            "ord": turn.get("ord", i),
                            "speaker": turn["speaker"],
                            "start_time_seconds": turn.get("start_time_seconds"),
                            "text": turn["text"],
                            "section_title": turn.get("section_title"),
                            "token_count": token_count,
                            "metadata": json.dumps(turn.get("metadata", {})),
                        },
                    )
                    result: dict | None = cur.fetchone()  # type: ignore[assignment]
                    if result:
                        turn_ids.append(result.get("id"))  # type: ignore[arg-type]
                conn.commit()

        return turn_ids

    def _insert_chunk_with_turn(
        self,
        doc_id: int,
        turn_id: int,
        ord: int,
        text: str,
        token_count: int,
    ) -> int | None:
        """Insert a single chunk with turn_id reference."""
        query = """
            INSERT INTO chunks (doc_id, turn_id, ord, text, token_count)
            VALUES (%(doc_id)s, %(turn_id)s, %(ord)s, %(text)s, %(token_count)s)
            RETURNING id
        """

        return execute_insert(
            query,
            {
                "doc_id": doc_id,
                "turn_id": turn_id,
                "ord": ord,
                "text": text,
                "token_count": token_count,
            },
        )
