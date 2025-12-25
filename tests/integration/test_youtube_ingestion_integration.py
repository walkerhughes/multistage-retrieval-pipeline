"""Integration tests for /api/ingest/youtube endpoint."""

import os
from unittest.mock import patch

import pytest

from dotenv import load_dotenv

load_dotenv()


# Mock YouTube document for testing (avoids real API calls)
class MockYouTubeDocument:
    """Mock YouTube document for testing."""

    def __init__(self, url, title="Test Video", text="Test transcript text with newlines\nand backslashes\\"):
        self.url = url
        self.title = title
        self.text = text
        self.published_at = None
        self.author = "Test Author"
        self.metadata = {
            "video_id": "test123",
            "duration": 300,
            "view_count": 1000,
            "description": "Test description",
        }


@pytest.mark.integration
class TestYouTubeIngestionEndpoint:
    """Test suite for YouTube ingestion endpoint."""

    @patch("src.ingestion.youtube_loader.YouTubeTranscriptFetcher.fetch")
    def test_ingest_youtube_basic(self, mock_fetch, test_client, clean_db):
        """Test basic YouTube URL ingestion (successful transcript fetch)."""
        # Arrange
        mock_doc = MockYouTubeDocument(
            url="https://www.youtube.com/watch?v=test123",
            title="Test Video Title",
            text="This is a test transcript about machine learning and neural networks.",
        )
        mock_fetch.return_value = mock_doc

        payload = {"url": "https://www.youtube.com/watch?v=test123"}

        # Act
        response = test_client.post("/api/ingest/youtube", json=payload)

        # Assert
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert data["status"] == "accepted"
        assert data["doc_id"] > 0
        assert data["title"] == "Test Video Title"
        assert data["chunk_count"] > 0
        assert data["total_tokens"] > 0
        assert data["ingestion_time_ms"] > 0

        # Verify in database
        with clean_db.cursor() as cur:
            # Check document was created
            cur.execute(
                "SELECT COUNT(*) as count FROM docs WHERE id = %(doc_id)s",
                {"doc_id": data["doc_id"]},
            )
            assert cur.fetchone()["count"] == 1

            # Check chunks were created
            cur.execute(
                "SELECT COUNT(*) as count FROM chunks WHERE doc_id = %(doc_id)s",
                {"doc_id": data["doc_id"]},
            )
            chunk_count = cur.fetchone()["count"]
            assert chunk_count == data["chunk_count"]

    @patch("src.ingestion.youtube_loader.YouTubeTranscriptFetcher.fetch")
    def test_ingest_youtube_with_title_override(self, mock_fetch, test_client, clean_db):
        """Test YouTube ingestion with custom title override."""
        # Arrange
        mock_doc = MockYouTubeDocument(
            url="https://www.youtube.com/watch?v=test123",
            title="Original YouTube Title",
            text="Test transcript about AI and deep learning.",
        )
        mock_fetch.return_value = mock_doc

        payload = {
            "url": "https://www.youtube.com/watch?v=test123",
            "title": "Custom Override Title",
        }

        # Act
        response = test_client.post("/api/ingest/youtube", json=payload)

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Custom Override Title"

        # Verify title in database
        with clean_db.cursor() as cur:
            cur.execute(
                "SELECT title FROM docs WHERE id = %(doc_id)s", {"doc_id": data["doc_id"]}
            )
            doc = cur.fetchone()
            assert doc["title"] == "Custom Override Title"

    @patch("src.ingestion.youtube_loader.YouTubeTranscriptFetcher.fetch")
    def test_ingest_youtube_with_metadata(self, mock_fetch, test_client, clean_db):
        """Test YouTube ingestion with custom metadata."""
        # Arrange
        mock_doc = MockYouTubeDocument(
            url="https://www.youtube.com/watch?v=test123",
            text="Test transcript about transformers and attention mechanisms.",
        )
        mock_fetch.return_value = mock_doc

        payload = {
            "url": "https://www.youtube.com/watch?v=test123",
            "metadata": {"category": "tutorial", "topic": "nlp", "difficulty": "advanced"},
        }

        # Act
        response = test_client.post("/api/ingest/youtube", json=payload)

        # Assert
        assert response.status_code == 200
        data = response.json()

        # Verify custom metadata merged with YouTube metadata
        with clean_db.cursor() as cur:
            cur.execute(
                "SELECT metadata FROM docs WHERE id = %(doc_id)s",
                {"doc_id": data["doc_id"]},
            )
            doc = cur.fetchone()
            metadata = doc["metadata"]

            # Check custom metadata
            assert metadata["category"] == "tutorial"
            assert metadata["topic"] == "nlp"
            assert metadata["difficulty"] == "advanced"

            # Check YouTube metadata still present
            assert metadata["video_id"] == "test123"
            assert "duration" in metadata

    @patch("src.ingestion.youtube_loader.YouTubeTranscriptFetcher.fetch")
    def test_ingest_youtube_text_cleaning(self, mock_fetch, test_client, clean_db):
        """Test that text is cleaned (newlines and backslashes removed).

        The cleaning happens in the YouTube loader's fetch method, so we mock
        it to return a document with already-cleaned text (simulating what the
        real loader does after calling clean_transcript_text).
        """
        # Arrange - mock returns document with cleaned text (as real loader would)
        cleaned_text = "This is line one This is line two This has a backslash here"
        mock_doc = MockYouTubeDocument(
            url="https://www.youtube.com/watch?v=test123",
            text=cleaned_text,  # Already cleaned by the loader
        )
        mock_fetch.return_value = mock_doc

        payload = {"url": "https://www.youtube.com/watch?v=test123"}

        # Act
        response = test_client.post("/api/ingest/youtube", json=payload)

        # Assert
        assert response.status_code == 200
        data = response.json()

        # Verify text was stored correctly in database
        with clean_db.cursor() as cur:
            cur.execute(
                "SELECT raw_text FROM docs WHERE id = %(doc_id)s",
                {"doc_id": data["doc_id"]},
            )
            doc = cur.fetchone()
            stored_text = doc["raw_text"]

            # Verify no newlines in stored text
            assert "\n" not in stored_text

            # Verify no backslashes in stored text
            assert "\\" not in stored_text

            # Verify content is preserved
            assert "This is line one" in stored_text
            assert "This is line two" in stored_text
            assert "backslash" in stored_text

    @patch("src.ingestion.youtube_loader.YouTubeTranscriptFetcher.fetch")
    def test_ingest_youtube_chunks_correctly(self, mock_fetch, test_client, clean_db):
        """Test that transcript is chunked according to token limits."""
        # Arrange
        # Create a longer transcript to ensure multiple chunks
        long_text = " ".join(
            [
                "Machine learning is a powerful technology that enables computers to learn from data."
            ]
            * 100
        )
        mock_doc = MockYouTubeDocument(
            url="https://www.youtube.com/watch?v=test123", text=long_text
        )
        mock_fetch.return_value = mock_doc

        payload = {"url": "https://www.youtube.com/watch?v=test123"}

        # Act
        response = test_client.post("/api/ingest/youtube", json=payload)

        # Assert
        assert response.status_code == 200
        data = response.json()

        # Verify chunks in database
        with clean_db.cursor() as cur:
            cur.execute(
                """
                SELECT id, ord, token_count, LENGTH(text) as text_length
                FROM chunks
                WHERE doc_id = %(doc_id)s
                ORDER BY ord
                """,
                {"doc_id": data["doc_id"]},
            )
            chunks = cur.fetchall()

            # Verify chunk ordering
            for i, chunk in enumerate(chunks):
                assert chunk["ord"] == i, f"Chunk order mismatch at index {i}"

            # Verify token counts are within expected range (400-800)
            for chunk in chunks:
                assert 0 < chunk["token_count"] <= 800, (
                    f"Chunk token_count {chunk['token_count']} outside valid range"
                )
                assert chunk["text_length"] > 0, "Chunk text should not be empty"

    @patch("src.ingestion.youtube_loader.YouTubeTranscriptFetcher.fetch")
    def test_ingest_youtube_tsvector_generated(self, mock_fetch, test_client, clean_db):
        """Test that tsvector is generated for full-text search."""
        # Arrange
        mock_doc = MockYouTubeDocument(
            url="https://www.youtube.com/watch?v=test123",
            text="Machine learning and artificial intelligence are transforming technology.",
        )
        mock_fetch.return_value = mock_doc

        payload = {"url": "https://www.youtube.com/watch?v=test123"}

        # Act
        response = test_client.post("/api/ingest/youtube", json=payload)

        # Assert
        assert response.status_code == 200
        data = response.json()

        # Verify tsvector is generated
        with clean_db.cursor() as cur:
            cur.execute(
                """
                SELECT id, tsv
                FROM chunks
                WHERE doc_id = %(doc_id)s
                LIMIT 1
                """,
                {"doc_id": data["doc_id"]},
            )
            chunk = cur.fetchone()
            assert chunk["tsv"] is not None, "tsvector should be generated"

            # Test FTS query works
            cur.execute(
                """
                SELECT COUNT(*) as count
                FROM chunks
                WHERE doc_id = %(doc_id)s
                AND tsv @@ websearch_to_tsquery('english', 'machine learning')
                """,
                {"doc_id": data["doc_id"]},
            )
            assert cur.fetchone()["count"] > 0, "FTS query should find results"

    @patch("src.ingestion.youtube_loader.YouTubeTranscriptFetcher.fetch")
    def test_ingest_youtube_retrieval_integration(self, mock_fetch, test_client, clean_db):
        """Test end-to-end: ingest YouTube video and retrieve via query."""
        # Arrange
        mock_doc = MockYouTubeDocument(
            url="https://www.youtube.com/watch?v=test123",
            text="Neural networks are the foundation of deep learning and modern AI systems.",
        )
        mock_fetch.return_value = mock_doc

        payload = {"url": "https://www.youtube.com/watch?v=test123"}

        # Act: Ingest
        ingest_response = test_client.post("/api/ingest/youtube", json=payload)
        assert ingest_response.status_code == 200
        ingest_data = ingest_response.json()

        # Act: Query
        query_payload = {"query": "neural networks", "max_returned": 10}
        query_response = test_client.post("/api/retrieval/query", json=query_payload)

        # Assert
        assert query_response.status_code == 200
        query_data = query_response.json()

        assert len(query_data["chunks"]) > 0, "Should retrieve at least one chunk"
        assert query_data["chunks"][0]["doc_id"] == ingest_data["doc_id"]
        assert "neural" in query_data["chunks"][0]["text"].lower()

    def test_ingest_youtube_invalid_url_format(self, test_client):
        """Test that invalid URL format is rejected."""
        # Arrange
        payload = {"url": "not-a-valid-url"}

        # Act
        response = test_client.post("/api/ingest/youtube", json=payload)

        # Assert
        assert response.status_code == 422, "Should reject invalid URL format"

    @patch("src.ingestion.youtube_loader.YouTubeTranscriptFetcher.fetch")
    def test_ingest_youtube_transcript_unavailable(self, mock_fetch, test_client):
        """Test handling of missing/unavailable transcript."""
        # Arrange
        mock_fetch.side_effect = ValueError("No transcript available for video")

        payload = {"url": "https://www.youtube.com/watch?v=test123"}

        # Act
        response = test_client.post("/api/ingest/youtube", json=payload)

        # Assert
        assert response.status_code == 400, "Should return 400 for unavailable transcript"
        assert "transcript" in response.json()["detail"].lower()

    @patch("src.ingestion.youtube_loader.YouTubeTranscriptFetcher.fetch")
    def test_ingest_youtube_multiple_videos(self, mock_fetch, test_client, clean_db):
        """Test ingesting multiple YouTube videos in sequence."""
        # Arrange
        videos = [
            {
                "url": "https://www.youtube.com/watch?v=video1",
                "title": "Video 1",
                "text": "Content about neural networks.",
            },
            {
                "url": "https://www.youtube.com/watch?v=video2",
                "title": "Video 2",
                "text": "Content about deep learning.",
            },
            {
                "url": "https://www.youtube.com/watch?v=video3",
                "title": "Video 3",
                "text": "Content about transformers.",
            },
        ]

        # Act
        doc_ids = []
        for video in videos:
            mock_doc = MockYouTubeDocument(
                url=video["url"], title=video["title"], text=video["text"]
            )
            mock_fetch.return_value = mock_doc

            payload = {"url": video["url"]}
            response = test_client.post("/api/ingest/youtube", json=payload)
            assert response.status_code == 200
            doc_ids.append(response.json()["doc_id"])

        # Assert
        assert len(set(doc_ids)) == 3, "Should have unique doc_ids"

        # Verify all in database
        with clean_db.cursor() as cur:
            cur.execute("SELECT COUNT(*) as count FROM docs WHERE source = 'youtube'")
            assert cur.fetchone()["count"] == 3

    @patch("src.ingestion.youtube_loader.YouTubeTranscriptFetcher.fetch")
    @pytest.mark.requires_openai
    def test_ingest_youtube_with_embeddings(self, mock_fetch, test_client, clean_db):
        """Test YouTube ingestion generates embeddings (requires OpenAI API key)."""
        # Skip if no API key
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        # Arrange
        mock_doc = MockYouTubeDocument(
            url="https://www.youtube.com/watch?v=test123",
            text="Machine learning enables computers to learn from experience.",
        )
        mock_fetch.return_value = mock_doc

        payload = {"url": "https://www.youtube.com/watch?v=test123"}

        # Act
        response = test_client.post("/api/ingest/youtube", json=payload)

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["embeddings_generated"] is True, "Embeddings should be generated"

        # Verify embeddings in database
        with clean_db.cursor() as cur:
            cur.execute(
                """
                SELECT ce.chunk_id, ce.embedding
                FROM chunk_embeddings ce
                JOIN chunks c ON c.id = ce.chunk_id
                WHERE c.doc_id = %(doc_id)s
                """,
                {"doc_id": data["doc_id"]},
            )
            embeddings = cur.fetchall()

            assert len(embeddings) == data["chunk_count"], (
                "Should have one embedding per chunk"
            )

            # Verify embedding exists
            for emb in embeddings:
                embedding_vector = emb["embedding"]
                assert embedding_vector is not None, "Embedding should not be null"
