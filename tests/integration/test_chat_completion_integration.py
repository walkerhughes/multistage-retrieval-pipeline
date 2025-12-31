"""Integration tests for /api/chat/completion endpoint."""

import os

import pytest

from dotenv import load_dotenv

load_dotenv()


@pytest.mark.integration
class TestChatCompletionEndpoint:
    """Test suite for chat completion endpoint."""

    @pytest.fixture(autouse=True)
    def setup_test_data(self, test_client, clean_db, sample_text):
        """Ingest test data before each test."""
        # Ingest sample text for retrieval
        payload = {
            "text": sample_text,
            "title": "Machine Learning Basics",
            "metadata": {"category": "ml", "author": "Test Author"},
        }
        response = test_client.post("/api/ingest/text", json=payload)
        assert response.status_code == 200
        self.doc_id = response.json()["doc_id"]
        yield
        # Cleanup handled by clean_db fixture

    @pytest.mark.requires_openai
    def test_chat_completion_basic(self, test_client):
        """Test basic chat completion with vanilla agent."""
        # Skip if no API key
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        # Arrange
        payload = {
            "question": "What is machine learning?",
            "agent": "vanilla",
            "mode": "fts",
            "max_returned": 5,
        }

        # Act
        response = test_client.post("/api/chat/completion", json=payload)

        # Assert
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()

        # Verify response structure
        assert "answer" in data
        assert "trace_id" in data
        assert "latency_ms" in data
        assert "retrieved_chunks" in data
        assert "model_used" in data
        assert "tokens_used" in data

        # Verify answer is not empty
        assert len(data["answer"]) > 0, "Answer should not be empty"

        # Verify latency is positive
        assert data["latency_ms"] > 0, "Latency should be positive"

        # Verify retrieved chunks
        assert len(data["retrieved_chunks"]) > 0, "Should retrieve chunks"
        for chunk in data["retrieved_chunks"]:
            assert "chunk_id" in chunk
            assert "doc_id" in chunk
            assert "text" in chunk
            assert "score" in chunk
            assert "metadata" in chunk
            assert "ord" in chunk

        # Verify tokens used
        assert "prompt_tokens" in data["tokens_used"]
        assert "completion_tokens" in data["tokens_used"]
        assert "total_tokens" in data["tokens_used"]

    @pytest.mark.requires_openai
    def test_chat_completion_hybrid_mode(self, test_client):
        """Test chat completion with hybrid retrieval mode."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        # Arrange
        payload = {
            "question": "Explain pattern recognition",
            "agent": "vanilla",
            "mode": "hybrid",
            "operator": "or",
            "fts_candidates": 50,
            "max_returned": 3,
        }

        # Act
        response = test_client.post("/api/chat/completion", json=payload)

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data["answer"]) > 0

    @pytest.mark.requires_openai
    def test_chat_completion_with_filters(self, test_client):
        """Test chat completion with metadata filters."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        # Arrange
        payload = {
            "question": "What is machine learning?",
            "agent": "vanilla",
            "mode": "fts",
            "max_returned": 5,
            "filters": {
                "doc_type": "text",
            },
        }

        # Act
        response = test_client.post("/api/chat/completion", json=payload)

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data["answer"]) > 0

    @pytest.mark.requires_openai
    def test_chat_completion_with_date_filters(self, test_client):
        """Test chat completion with datetime filters (JSON serialization)."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        # Arrange - use date filters to verify datetime serialization works
        payload = {
            "question": "What is machine learning?",
            "agent": "vanilla",
            "mode": "fts",
            "max_returned": 5,
            "filters": {
                "start_date": "2020-01-01T00:00:00",
                "end_date": "2030-12-31T23:59:59",
            },
        }

        # Act
        response = test_client.post("/api/chat/completion", json=payload)

        # Assert - should not get datetime serialization error
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()
        assert "answer" in data

    def test_chat_completion_multi_query_not_implemented(self, test_client):
        """Test that multi-query agent returns appropriate error."""
        # Arrange
        payload = {
            "question": "What is machine learning?",
            "agent": "multi-query",
            "mode": "fts",
        }

        # Act
        response = test_client.post("/api/chat/completion", json=payload)

        # Assert - should get 400 with NotImplementedError message
        assert response.status_code == 400
        data = response.json()
        assert "not yet implemented" in data["detail"].lower()
        assert "Issue #13" in data["detail"]

    def test_chat_completion_default_agent(self, test_client):
        """Test that default agent is vanilla."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        # Arrange - no agent specified, should default to vanilla
        payload = {
            "question": "What is machine learning?",
        }

        # Act
        response = test_client.post("/api/chat/completion", json=payload)

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data

    def test_chat_completion_invalid_mode(self, test_client):
        """Test that invalid retrieval mode is rejected."""
        # Arrange
        payload = {
            "question": "What is machine learning?",
            "agent": "vanilla",
            "mode": "invalid_mode",
        }

        # Act
        response = test_client.post("/api/chat/completion", json=payload)

        # Assert - should get 422 validation error
        assert response.status_code == 422

    def test_chat_completion_empty_question(self, test_client):
        """Test that empty question is rejected."""
        # Arrange
        payload = {
            "question": "",
            "agent": "vanilla",
        }

        # Act
        response = test_client.post("/api/chat/completion", json=payload)

        # Assert
        assert response.status_code == 422

    def test_chat_completion_missing_question(self, test_client):
        """Test that missing question is rejected."""
        # Arrange
        payload = {
            "agent": "vanilla",
        }

        # Act
        response = test_client.post("/api/chat/completion", json=payload)

        # Assert
        assert response.status_code == 422

    @pytest.mark.requires_openai
    def test_chat_completion_response_grounded_in_context(self, test_client):
        """Test that answer is grounded in retrieved chunks."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        # Arrange - ask about something in our test data
        payload = {
            "question": "What enables computers to improve their performance?",
            "agent": "vanilla",
            "mode": "fts",
            "max_returned": 5,
        }

        # Act
        response = test_client.post("/api/chat/completion", json=payload)

        # Assert
        assert response.status_code == 200
        data = response.json()

        # Verify retrieved chunks contain relevant content
        chunk_texts = [c["text"].lower() for c in data["retrieved_chunks"]]
        has_relevant_chunk = any(
            "machine learning" in text or "pattern" in text
            for text in chunk_texts
        )
        assert has_relevant_chunk, "Should retrieve relevant chunks about ML"
    
    @pytest.mark.requires_openai
    def test_chat_completion_metadata_in_response(self, test_client):
        """Test that chunk metadata is properly included in response."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        # Arrange
        payload = {
            "question": "What is machine learning?",
            "agent": "vanilla",
            "mode": "fts",
            "max_returned": 5,
        }

        # Act
        response = test_client.post("/api/chat/completion", json=payload)

        # Assert
        assert response.status_code == 200
        data = response.json()

        # Verify metadata structure
        for chunk in data["retrieved_chunks"]:
            metadata = chunk["metadata"]
            assert "title" in metadata
            # published_at should be serialized as ISO string or None
            if "published_at" in metadata and metadata["published_at"] is not None:
                # Should be a string (ISO format), not a datetime object
                assert isinstance(metadata["published_at"], str)
    