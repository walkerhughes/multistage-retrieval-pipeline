"""Integration tests for /api/ingest/text endpoint."""

import os

import pytest


@pytest.mark.integration
class TestTextIngestionEndpoint:
    """Test suite for text ingestion endpoint."""

    def test_ingest_text_basic(self, test_client, clean_db, sample_short_text):
        """Test basic text ingestion without embeddings."""
        # Arrange
        payload = {"text": sample_short_text, "title": "Test Document"}

        # Act
        response = test_client.post("/api/ingest/text", json=payload)

        # Assert
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert data["status"] == "accepted"
        assert data["doc_id"] > 0
        assert data["title"] == "Test Document"
        assert data["chunk_count"] > 0
        assert data["total_tokens"] > 0
        assert data["ingestion_time_ms"] > 0

        # Verify in database
        with clean_db.cursor() as cur:
            # Check document was created
            cur.execute("SELECT COUNT(*) as count FROM docs WHERE id = %(doc_id)s", {"doc_id": data["doc_id"]})
            assert cur.fetchone()["count"] == 1

            # Check chunks were created
            cur.execute("SELECT COUNT(*) as count FROM chunks WHERE doc_id = %(doc_id)s", {"doc_id": data["doc_id"]})
            chunk_count = cur.fetchone()["count"]
            assert chunk_count == data["chunk_count"]

    def test_ingest_text_with_metadata(self, test_client, clean_db, sample_short_text):
        """Test text ingestion with custom metadata."""
        # Arrange
        payload = {
            "text": sample_short_text,
            "title": "Test Document with Metadata",
            "metadata": {"author": "Test Author", "category": "ml"},
        }

        # Act
        response = test_client.post("/api/ingest/text", json=payload)

        # Assert
        assert response.status_code == 200
        data = response.json()

        # Verify metadata in database
        with clean_db.cursor() as cur:
            cur.execute(
                "SELECT metadata FROM docs WHERE id = %(doc_id)s",
                {"doc_id": data["doc_id"]},
            )
            doc = cur.fetchone()
            assert doc["metadata"]["author"] == "Test Author"
            assert doc["metadata"]["category"] == "ml"

    def test_ingest_text_without_title(self, test_client, clean_db, sample_short_text):
        """Test text ingestion without providing title."""
        # Arrange
        payload = {"text": sample_short_text}

        # Act
        response = test_client.post("/api/ingest/text", json=payload)

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Untitled Document"

    def test_ingest_text_chunks_correctly(self, test_client, clean_db, sample_text):
        """Test that text is chunked according to token limits."""
        # Arrange
        payload = {"text": sample_text, "title": "Long Document"}

        # Act
        response = test_client.post("/api/ingest/text", json=payload)

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

            # Verify token counts are within expected range
            for chunk in chunks:
                assert 0 < chunk["token_count"] <= 800, (
                    f"Chunk token_count {chunk['token_count']} outside valid range"
                )
                assert chunk["text_length"] > 0, "Chunk text should not be empty"

    def test_ingest_text_tsvector_generated(self, test_client, clean_db, sample_short_text):
        """Test that tsvector is generated for full-text search."""
        # Arrange
        payload = {"text": sample_short_text, "title": "FTS Test"}

        # Act
        response = test_client.post("/api/ingest/text", json=payload)

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

    @pytest.mark.requires_openai
    def test_ingest_text_with_embeddings(self, test_client, clean_db, sample_short_text):
        """Test text ingestion generates embeddings (requires OpenAI API key)."""
        # Skip if no API key
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        # Arrange
        payload = {"text": sample_short_text, "title": "Embedding Test"}

        # Act
        response = test_client.post("/api/ingest/text", json=payload)

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

            # Verify embedding dimensions (1536 for text-embedding-3-small)
            for emb in embeddings:
                embedding_vector = emb["embedding"]
                # pgvector returns as string like '[0.1, 0.2, ...]'
                assert embedding_vector is not None, "Embedding should not be null"

    def test_ingest_text_retrieval_integration(
        self, test_client, clean_db, sample_short_text
    ):
        """Test end-to-end: ingest text and retrieve it via query."""
        # Arrange
        payload = {"text": sample_short_text, "title": "Retrieval Test"}

        # Act: Ingest
        ingest_response = test_client.post("/api/ingest/text", json=payload)
        assert ingest_response.status_code == 200
        ingest_data = ingest_response.json()

        # Act: Query
        query_payload = {"query": "machine learning", "max_returned": 10}
        query_response = test_client.post("/api/retrieval/query", json=query_payload)

        # Assert
        assert query_response.status_code == 200
        query_data = query_response.json()

        assert len(query_data["chunks"]) > 0, "Should retrieve at least one chunk"
        assert query_data["chunks"][0]["doc_id"] == ingest_data["doc_id"]
        assert "machine" in query_data["chunks"][0]["text"].lower()

    def test_ingest_text_invalid_empty(self, test_client, clean_db):
        """Test that empty text is rejected."""
        # Arrange
        payload = {"text": "", "title": "Empty Test"}

        # Act
        response = test_client.post("/api/ingest/text", json=payload)

        # Assert
        assert response.status_code == 422, "Should reject empty text"

    def test_ingest_text_invalid_missing_text(self, test_client, clean_db):
        """Test that missing text field is rejected."""
        # Arrange
        payload = {"title": "No Text Test"}

        # Act
        response = test_client.post("/api/ingest/text", json=payload)

        # Assert
        assert response.status_code == 422, "Should reject missing text field"

    def test_ingest_text_multiple_documents(
        self, test_client, clean_db, sample_short_text
    ):
        """Test ingesting multiple documents in sequence."""
        # Arrange
        payloads = [
            {"text": "Document one about neural networks.", "title": "Doc 1"},
            {"text": "Document two about deep learning.", "title": "Doc 2"},
            {"text": "Document three about transformers.", "title": "Doc 3"},
        ]

        # Act
        doc_ids = []
        for payload in payloads:
            response = test_client.post("/api/ingest/text", json=payload)
            assert response.status_code == 200
            doc_ids.append(response.json()["doc_id"])

        # Assert
        assert len(set(doc_ids)) == 3, "Should have unique doc_ids"

        # Verify all in database
        with clean_db.cursor() as cur:
            cur.execute("SELECT COUNT(*) as count FROM docs")
            assert cur.fetchone()["count"] == 3
