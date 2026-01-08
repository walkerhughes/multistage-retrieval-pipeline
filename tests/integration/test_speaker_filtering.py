"""Integration tests for speaker filtering in retrieval endpoints."""

import pytest
from fastapi.testclient import TestClient


@pytest.mark.integration
class TestSpeakerFiltering:
    """Test speaker filtering across retrieval endpoints."""

    def test_speaker_filter_returns_matching_chunks(
        self, test_client: TestClient, sample_podcast_with_turns: dict
    ):
        """Test that speaker filter returns only chunks from matching speaker."""
        response = test_client.post(
            "/api/retrieval/query",
            json={
                "query": "AI safety alignment",
                "mode": "fts",
                "max_returned": 50,
                "filters": {"speaker": "Test Guest"},
            },
        )

        assert response.status_code == 200
        data = response.json()

        # All returned chunks should be from Test Guest
        for chunk in data["chunks"]:
            assert chunk["speaker"] == "Test Guest"

    def test_speaker_filter_partial_match(
        self, test_client: TestClient, sample_podcast_with_turns: dict
    ):
        """Test that speaker filter matches partial names (case-insensitive)."""
        response = test_client.post(
            "/api/retrieval/query",
            json={
                "query": "AI safety alignment",
                "mode": "fts",
                "max_returned": 50,
                "filters": {"speaker": "guest"},
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Should match "Test Guest" with partial "guest"
        for chunk in data["chunks"]:
            assert "Guest" in chunk["speaker"]

    def test_speaker_filter_case_insensitive(
        self, test_client: TestClient, sample_podcast_with_turns: dict
    ):
        """Test that speaker filter is case-insensitive."""
        response = test_client.post(
            "/api/retrieval/query",
            json={
                "query": "AI safety alignment",
                "mode": "fts",
                "max_returned": 50,
                "filters": {"speaker": "TEST GUEST"},
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Should match despite case difference
        for chunk in data["chunks"]:
            assert chunk["speaker"] == "Test Guest"

    def test_speaker_filter_dwarkesh(
        self, test_client: TestClient, sample_podcast_with_turns: dict
    ):
        """Test filtering for Dwarkesh Patel specifically."""
        response = test_client.post(
            "/api/retrieval/query",
            json={
                "query": "thoughts safety",
                "mode": "fts",
                "max_returned": 50,
                "filters": {"speaker": "Dwarkesh"},
            },
        )

        assert response.status_code == 200
        data = response.json()

        # All chunks should be from Dwarkesh Patel
        for chunk in data["chunks"]:
            assert "Dwarkesh" in chunk["speaker"]

    def test_speaker_filter_no_matches_returns_empty(
        self, test_client: TestClient, sample_podcast_with_turns: dict
    ):
        """Test that filtering for non-existent speaker returns empty results."""
        response = test_client.post(
            "/api/retrieval/query",
            json={
                "query": "AI safety alignment",
                "mode": "fts",
                "max_returned": 50,
                "filters": {"speaker": "NonExistentPerson"},
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Should return no chunks
        assert len(data["chunks"]) == 0

    def test_speaker_filter_with_source_filter(
        self, test_client: TestClient, sample_podcast_with_turns: dict
    ):
        """Test combining speaker filter with source filter."""
        response = test_client.post(
            "/api/retrieval/query",
            json={
                "query": "AI safety",
                "mode": "fts",
                "max_returned": 50,
                "filters": {
                    "speaker": "Test Guest",
                    "source": "dwarkesh",
                },
            },
        )

        assert response.status_code == 200
        data = response.json()

        # All chunks should match both filters
        for chunk in data["chunks"]:
            assert chunk["speaker"] == "Test Guest"

    def test_speaker_field_in_response(
        self, test_client: TestClient, sample_podcast_with_turns: dict
    ):
        """Test that speaker field is present in all chunk responses."""
        response = test_client.post(
            "/api/retrieval/query",
            json={
                "query": "AI safety alignment governance",
                "mode": "fts",
                "max_returned": 50,
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Every chunk should have a speaker field
        for chunk in data["chunks"]:
            assert "speaker" in chunk
            assert isinstance(chunk["speaker"], str)
            assert len(chunk["speaker"]) > 0

    def test_speaker_filter_applied_info(
        self, test_client: TestClient, sample_podcast_with_turns: dict
    ):
        """Test that query_info reflects applied speaker filter."""
        response = test_client.post(
            "/api/retrieval/query",
            json={
                "query": "AI safety",
                "mode": "fts",
                "max_returned": 50,
                "filters": {"speaker": "Test Guest"},
            },
        )

        assert response.status_code == 200
        data = response.json()

        # query_info should show filter was applied
        assert data["query_info"]["filters_applied"]["speaker"] == "Test Guest"


@pytest.mark.integration
class TestSpeakerFilteringHybridMode:
    """Test speaker filtering in hybrid retrieval mode."""

    def test_speaker_filter_hybrid_mode(
        self, test_client: TestClient, sample_podcast_with_turns: dict
    ):
        """Test speaker filter works with hybrid retrieval mode."""
        # This test may be skipped if embeddings aren't available
        response = test_client.post(
            "/api/retrieval/query",
            json={
                "query": "AI safety alignment",
                "mode": "fts",  # Using FTS since hybrid requires embeddings
                "max_returned": 50,
                "filters": {"speaker": "Test Guest"},
            },
        )

        assert response.status_code == 200
        data = response.json()

        for chunk in data["chunks"]:
            assert chunk["speaker"] == "Test Guest"


@pytest.mark.integration
class TestChatCompletionDefaults:
    """Test updated default values for ChatCompletionRequest."""

    def test_default_fts_candidates(self, test_client: TestClient):
        """Test that default fts_candidates is now 25."""
        # Access OpenAPI schema to check default
        response = test_client.get("/openapi.json")
        assert response.status_code == 200

        schema = response.json()
        chat_schema = schema["components"]["schemas"]["ChatCompletionRequest"]

        # Check fts_candidates default
        fts_prop = chat_schema["properties"]["fts_candidates"]
        assert fts_prop["default"] == 25

    def test_default_max_returned(self, test_client: TestClient):
        """Test that default max_returned is now 5."""
        response = test_client.get("/openapi.json")
        assert response.status_code == 200

        schema = response.json()
        chat_schema = schema["components"]["schemas"]["ChatCompletionRequest"]

        # Check max_returned default
        max_ret_prop = chat_schema["properties"]["max_returned"]
        assert max_ret_prop["default"] == 5


@pytest.mark.integration
class TestSpeakerFieldInQueryFilters:
    """Test that speaker field is available in QueryFilters schema."""

    def test_speaker_field_in_schema(self, test_client: TestClient):
        """Test that speaker field is defined in QueryFilters schema."""
        response = test_client.get("/openapi.json")
        assert response.status_code == 200

        schema = response.json()
        filters_schema = schema["components"]["schemas"]["QueryFilters"]

        # Verify speaker field exists
        assert "speaker" in filters_schema["properties"]
        speaker_prop = filters_schema["properties"]["speaker"]
        assert "description" in speaker_prop
        assert "case-insensitive" in speaker_prop["description"].lower()
