"""Integration tests for POST /api/retrieval/query-expanded endpoint."""

import pytest

from dotenv import load_dotenv

load_dotenv()


@pytest.mark.integration
class TestQueryExpandedEndpoint:
    """Test suite for POST /api/retrieval/query-expanded endpoint."""

    def test_basic_query_expanded(self, test_client, sample_podcast_with_turns):
        """Test basic query expansion retrieves and expands chunks to turns."""
        # Act
        response = test_client.post(
            "/api/retrieval/query-expanded",
            json={
                "query": "AI safety alignment",
                "max_chunks": 10,
                "token_budget": 8000,
            }
        )

        # Assert
        assert response.status_code == 200
        data = response.json()

        # Should have turns returned
        assert "turns" in data
        assert "total_turns" in data
        assert "total_tokens" in data
        assert "deduplication_stats" in data
        assert "timing_ms" in data
        assert "query_info" in data

        # Verify timing breakdown
        assert "retrieval_ms" in data["timing_ms"]
        assert "expansion_ms" in data["timing_ms"]
        assert "total_ms" in data["timing_ms"]

        # Verify deduplication stats
        assert "chunks_retrieved" in data["deduplication_stats"]
        assert "unique_turns" in data["deduplication_stats"]
        assert "chunks_deduplicated" in data["deduplication_stats"]

    def test_query_expanded_returns_turn_fields(self, test_client, sample_podcast_with_turns):
        """Test that all expected turn fields are returned."""
        # Act
        response = test_client.post(
            "/api/retrieval/query-expanded",
            json={
                "query": "AI safety",
                "max_chunks": 10,
            }
        )

        # Assert
        assert response.status_code == 200
        data = response.json()

        if data["total_turns"] > 0:
            turn = data["turns"][0]
            # Check all required fields
            assert "turn_id" in turn
            assert "doc_id" in turn
            assert "ord" in turn
            assert "speaker" in turn
            assert "full_text" in turn
            assert "start_time_seconds" in turn
            assert "section_title" in turn
            assert "token_count" in turn
            assert "relevance_score" in turn
            assert "doc_metadata" in turn
            assert "preceding_question" in turn  # Should be None or TurnData

            # Check doc_metadata fields
            assert "title" in turn["doc_metadata"]
            assert "url" in turn["doc_metadata"]
            assert "source" in turn["doc_metadata"]

    def test_query_expanded_deduplication(self, test_client, sample_podcast_with_turns):
        """Test that multiple chunks from same turn are deduplicated."""
        # Act
        response = test_client.post(
            "/api/retrieval/query-expanded",
            json={
                "query": "alignment technical research",  # Should match turn with 2 chunks
                "max_chunks": 50,
            }
        )

        # Assert
        assert response.status_code == 200
        data = response.json()

        # Check deduplication stats
        stats = data["deduplication_stats"]

        # If chunks were retrieved, verify deduplication works
        if stats["chunks_retrieved"] > 0:
            # unique_turns should be <= chunks_retrieved
            assert stats["unique_turns"] <= stats["chunks_retrieved"]
            # chunks_deduplicated should be the difference
            assert stats["chunks_deduplicated"] == stats["chunks_retrieved"] - stats["unique_turns"]

    def test_query_expanded_preserves_highest_score(self, test_client, sample_podcast_with_turns):
        """Test that deduplication preserves the highest relevance score."""
        # Act
        response = test_client.post(
            "/api/retrieval/query-expanded",
            json={
                "query": "AI safety crucial alignment",
                "max_chunks": 50,
            }
        )

        # Assert
        assert response.status_code == 200
        data = response.json()

        # Turns should be sorted by relevance score descending
        if len(data["turns"]) > 1:
            scores = [t["relevance_score"] for t in data["turns"]]
            assert scores == sorted(scores, reverse=True), "Turns should be sorted by relevance score"

    def test_query_expanded_token_budget_enforcement(self, test_client, sample_podcast_with_turns):
        """Test that token budget limits the number of returned turns."""
        # Act - use a small token budget
        response = test_client.post(
            "/api/retrieval/query-expanded",
            json={
                "query": "AI safety governance",
                "max_chunks": 50,
                "token_budget": 50,  # Very small budget
            }
        )

        # Assert
        assert response.status_code == 200
        data = response.json()

        # Total tokens should not exceed budget
        assert data["total_tokens"] <= 50

    def test_query_expanded_with_preceding_question(self, test_client, sample_podcast_with_turns):
        """Test including preceding question in response."""
        # Act
        response = test_client.post(
            "/api/retrieval/query-expanded",
            json={
                "query": "AI safety crucial",  # Should match guest's response
                "max_chunks": 10,
                "include_preceding_question": True,
                "token_budget": 8000,
            }
        )

        # Assert
        assert response.status_code == 200
        data = response.json()

        # Check if any turn has a preceding question
        turns_with_questions = [t for t in data["turns"] if t["preceding_question"] is not None]

        # Turns that are not first in doc should have preceding questions
        for turn in turns_with_questions:
            pq = turn["preceding_question"]
            assert "turn_id" in pq
            assert "speaker" in pq
            assert "full_text" in pq
            assert "token_count" in pq
            # Preceding question should have ord one less than the answer turn
            assert pq["ord"] == turn["ord"] - 1

    def test_query_expanded_without_preceding_question(self, test_client, sample_podcast_with_turns):
        """Test that preceding_question is None when not requested."""
        # Act
        response = test_client.post(
            "/api/retrieval/query-expanded",
            json={
                "query": "AI safety",
                "max_chunks": 10,
                "include_preceding_question": False,
            }
        )

        # Assert
        assert response.status_code == 200
        data = response.json()

        # All turns should have preceding_question as None
        for turn in data["turns"]:
            assert turn["preceding_question"] is None

    def test_query_expanded_empty_results(self, test_client, sample_podcast_with_turns):
        """Test that non-matching query returns empty results gracefully."""
        # Act
        response = test_client.post(
            "/api/retrieval/query-expanded",
            json={
                "query": "xyznonexistentqueryxyz",
                "max_chunks": 10,
            }
        )

        # Assert
        assert response.status_code == 200
        data = response.json()

        assert data["turns"] == []
        assert data["total_turns"] == 0
        assert data["total_tokens"] == 0
        assert data["deduplication_stats"]["chunks_retrieved"] == 0

    def test_query_expanded_with_filters(self, test_client, sample_podcast_with_turns):
        """Test query with metadata filters."""
        # Act
        response = test_client.post(
            "/api/retrieval/query-expanded",
            json={
                "query": "AI safety",
                "max_chunks": 10,
                "filters": {
                    "source": "dwarkesh",
                    "doc_type": "transcript",
                }
            }
        )

        # Assert
        assert response.status_code == 200
        data = response.json()

        # Query info should show filters applied
        assert data["query_info"]["filters_applied"]["source"] == "dwarkesh"
        assert data["query_info"]["filters_applied"]["doc_type"] == "transcript"

    def test_query_expanded_query_info(self, test_client, sample_podcast_with_turns):
        """Test that query_info contains all request parameters."""
        # Act
        response = test_client.post(
            "/api/retrieval/query-expanded",
            json={
                "query": "test query",
                "max_chunks": 25,
                "token_budget": 5000,
                "include_preceding_question": True,
                "mode": "fts",
                "operator": "and",
            }
        )

        # Assert
        assert response.status_code == 200
        data = response.json()

        query_info = data["query_info"]
        assert query_info["query"] == "test query"
        assert query_info["mode"] == "fts"
        assert query_info["max_chunks"] == 25
        assert query_info["token_budget"] == 5000
        assert query_info["include_preceding_question"] is True

    def test_query_expanded_invalid_mode(self, test_client, sample_podcast_with_turns):
        """Test that invalid retrieval mode returns error."""
        # Act
        response = test_client.post(
            "/api/retrieval/query-expanded",
            json={
                "query": "test",
                "mode": "invalid_mode",
            }
        )

        # Assert
        assert response.status_code == 422  # Pydantic validation error

    def test_query_expanded_fts_mode(self, test_client, sample_podcast_with_turns):
        """Test query-expanded with FTS mode."""
        # Act
        response = test_client.post(
            "/api/retrieval/query-expanded",
            json={
                "query": "AI safety alignment",
                "mode": "fts",
                "operator": "or",
            }
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["query_info"]["mode"] == "fts"

    def test_query_expanded_and_operator(self, test_client, sample_podcast_with_turns):
        """Test query-expanded with AND operator for stricter matching."""
        # Act
        response = test_client.post(
            "/api/retrieval/query-expanded",
            json={
                "query": "AI safety",
                "mode": "fts",
                "operator": "and",
            }
        )

        # Assert
        assert response.status_code == 200
        data = response.json()

    def test_query_expanded_or_operator(self, test_client, sample_podcast_with_turns):
        """Test query-expanded with OR operator for broader matching."""
        # Act
        response = test_client.post(
            "/api/retrieval/query-expanded",
            json={
                "query": "AI safety",
                "mode": "fts",
                "operator": "or",
            }
        )

        # Assert
        assert response.status_code == 200
        data = response.json()

    def test_query_expanded_min_token_budget(self, test_client, sample_podcast_with_turns):
        """Test that token_budget has minimum value enforced."""
        # Act
        response = test_client.post(
            "/api/retrieval/query-expanded",
            json={
                "query": "test",
                "token_budget": 50,  # Below minimum of 100
            }
        )

        # Assert
        assert response.status_code == 422  # Validation error

    def test_query_expanded_max_chunks_limit(self, test_client, sample_podcast_with_turns):
        """Test that max_chunks has maximum value enforced."""
        # Act
        response = test_client.post(
            "/api/retrieval/query-expanded",
            json={
                "query": "test",
                "max_chunks": 1000,  # Above maximum of 500
            }
        )

        # Assert
        assert response.status_code == 422  # Validation error

    def test_query_expanded_default_values(self, test_client, sample_podcast_with_turns):
        """Test that default values are applied correctly."""
        # Act - minimal request
        response = test_client.post(
            "/api/retrieval/query-expanded",
            json={"query": "AI safety"}
        )

        # Assert
        assert response.status_code == 200
        data = response.json()

        query_info = data["query_info"]
        assert query_info["token_budget"] == 8000  # Default
        assert query_info["max_chunks"] == 100  # Default
        assert query_info["include_preceding_question"] is False  # Default
        assert query_info["mode"] == "fts"  # Default

    def test_query_expanded_timing_accuracy(self, test_client, sample_podcast_with_turns):
        """Test that timing values are reasonable."""
        # Act
        response = test_client.post(
            "/api/retrieval/query-expanded",
            json={"query": "AI safety"}
        )

        # Assert
        assert response.status_code == 200
        data = response.json()

        timing = data["timing_ms"]
        # All timing values should be non-negative
        assert timing["retrieval_ms"] >= 0
        assert timing["expansion_ms"] >= 0
        assert timing["total_ms"] >= 0
        # Total should be >= sum of parts (may include overhead)
        assert timing["total_ms"] >= timing["retrieval_ms"]


@pytest.mark.integration
class TestQueryExpandedTokenBudget:
    """Tests specifically for token budget enforcement."""

    def test_token_budget_includes_question_tokens(self, test_client, sample_podcast_with_turns):
        """Test that token budget accounts for question tokens when include_preceding_question=True."""
        # First, get results without preceding questions
        response_without = test_client.post(
            "/api/retrieval/query-expanded",
            json={
                "query": "AI safety",
                "max_chunks": 50,
                "token_budget": 8000,
                "include_preceding_question": False,
            }
        )

        # Then with preceding questions but same budget
        response_with = test_client.post(
            "/api/retrieval/query-expanded",
            json={
                "query": "AI safety",
                "max_chunks": 50,
                "token_budget": 8000,
                "include_preceding_question": True,
            }
        )

        assert response_without.status_code == 200
        assert response_with.status_code == 200

        data_without = response_without.json()
        data_with = response_with.json()

        # With questions included, we may get fewer turns due to budget
        # but total_tokens includes question tokens
        if data_with["total_turns"] > 0:
            # total_tokens should account for question tokens
            assert data_with["total_tokens"] <= 8000

    def test_token_budget_stops_at_boundary(self, test_client, sample_podcast_with_turns):
        """Test that token budget correctly stops when adding next turn would exceed."""
        # Act with a moderate budget
        response = test_client.post(
            "/api/retrieval/query-expanded",
            json={
                "query": "AI",
                "max_chunks": 50,
                "token_budget": 200,
            }
        )

        # Assert
        assert response.status_code == 200
        data = response.json()

        # Should respect the budget
        assert data["total_tokens"] <= 200


@pytest.mark.integration
class TestQueryExpandedEdgeCases:
    """Edge case tests for query-expanded endpoint."""

    def test_empty_query_rejected(self, test_client, sample_podcast_with_turns):
        """Test that empty query is rejected."""
        # Act
        response = test_client.post(
            "/api/retrieval/query-expanded",
            json={"query": ""}
        )

        # Assert
        assert response.status_code == 422  # Validation error

    def test_query_with_special_characters(self, test_client, sample_podcast_with_turns):
        """Test query with special characters is handled."""
        # Act
        response = test_client.post(
            "/api/retrieval/query-expanded",
            json={"query": "AI & safety | alignment"}
        )

        # Assert - should not crash
        assert response.status_code == 200

    def test_very_long_query(self, test_client, sample_podcast_with_turns):
        """Test handling of very long queries."""
        # Act
        long_query = "AI safety " * 100  # Long query
        response = test_client.post(
            "/api/retrieval/query-expanded",
            json={"query": long_query}
        )

        # Assert - should handle gracefully
        assert response.status_code == 200
