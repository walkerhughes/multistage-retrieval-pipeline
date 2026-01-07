"""Integration tests for chunk expansion endpoints."""

import pytest

from dotenv import load_dotenv

load_dotenv()


@pytest.mark.integration
class TestExpandChunksEndpoint:
    """Test suite for POST /api/retrieval/expand endpoint."""

    def test_expand_single_chunk(self, test_client, sample_podcast_with_turns):
        """Test expanding a single chunk to its parent turn."""
        # Arrange
        chunk_ids = [sample_podcast_with_turns["chunk_ids"][0]]

        # Act
        response = test_client.post("/api/retrieval/expand", json={"chunk_ids": chunk_ids})

        # Assert
        assert response.status_code == 200
        data = response.json()

        assert data["total_turns"] == 1
        assert len(data["turns"]) == 1
        assert data["query_time_ms"] >= 0

        turn = data["turns"][0]
        assert turn["turn_id"] == sample_podcast_with_turns["turn_ids"][0]
        assert turn["speaker"] == "Dwarkesh Patel"
        assert turn["full_text"] == "What are your thoughts on AI safety?"
        assert turn["section_title"] == "Introduction"
        assert turn["start_time_seconds"] == 0

    def test_expand_multiple_chunks_different_turns(self, test_client, sample_podcast_with_turns):
        """Test expanding multiple chunks from different turns."""
        # Arrange - chunks 0 and 3 are from different turns
        chunk_ids = [
            sample_podcast_with_turns["chunk_ids"][0],
            sample_podcast_with_turns["chunk_ids"][3],
        ]

        # Act
        response = test_client.post("/api/retrieval/expand", json={"chunk_ids": chunk_ids})

        # Assert
        assert response.status_code == 200
        data = response.json()

        assert data["total_turns"] == 2
        assert len(data["turns"]) == 2

        # Verify turns are ordered by doc_id, ord
        speakers = [t["speaker"] for t in data["turns"]]
        assert speakers == ["Dwarkesh Patel", "Dwarkesh Patel"]  # ord 0 and ord 2

    def test_expand_deduplicates_same_turn(self, test_client, sample_podcast_with_turns):
        """Test that multiple chunks from the same turn return only one turn."""
        # Arrange - chunks 1 and 2 are both from turn_ids[1]
        chunk_ids = [
            sample_podcast_with_turns["chunk_ids"][1],
            sample_podcast_with_turns["chunk_ids"][2],
        ]

        # Act
        response = test_client.post("/api/retrieval/expand", json={"chunk_ids": chunk_ids})

        # Assert
        assert response.status_code == 200
        data = response.json()

        # Should be deduplicated to 1 turn
        assert data["total_turns"] == 1
        assert len(data["turns"]) == 1

        turn = data["turns"][0]
        assert turn["turn_id"] == sample_podcast_with_turns["turn_ids"][1]
        assert turn["speaker"] == "Test Guest"
        assert "AI safety is crucial" in turn["full_text"]

    def test_expand_nonexistent_chunk_ids(self, test_client, sample_podcast_with_turns):
        """Test expanding with non-existent chunk IDs returns empty result."""
        # Arrange
        chunk_ids = [999999, 888888]

        # Act
        response = test_client.post("/api/retrieval/expand", json={"chunk_ids": chunk_ids})

        # Assert
        assert response.status_code == 200
        data = response.json()

        assert data["total_turns"] == 0
        assert len(data["turns"]) == 0

    def test_expand_mixed_valid_invalid_chunk_ids(self, test_client, sample_podcast_with_turns):
        """Test expanding with mix of valid and invalid chunk IDs."""
        # Arrange
        chunk_ids = [
            sample_podcast_with_turns["chunk_ids"][0],
            999999,  # Invalid
        ]

        # Act
        response = test_client.post("/api/retrieval/expand", json={"chunk_ids": chunk_ids})

        # Assert
        assert response.status_code == 200
        data = response.json()

        # Should return only the valid turn
        assert data["total_turns"] == 1
        assert len(data["turns"]) == 1

    def test_expand_empty_chunk_ids_rejected(self, test_client, sample_podcast_with_turns):
        """Test that empty chunk_ids list is rejected by validation."""
        # Arrange
        chunk_ids: list[int] = []

        # Act
        response = test_client.post("/api/retrieval/expand", json={"chunk_ids": chunk_ids})

        # Assert
        assert response.status_code == 422  # Pydantic validation error

    def test_expand_returns_correct_turn_fields(self, test_client, sample_podcast_with_turns):
        """Test that all turn fields are returned correctly."""
        # Arrange
        chunk_ids = [sample_podcast_with_turns["chunk_ids"][1]]

        # Act
        response = test_client.post("/api/retrieval/expand", json={"chunk_ids": chunk_ids})

        # Assert
        assert response.status_code == 200
        data = response.json()

        turn = data["turns"][0]
        assert "turn_id" in turn
        assert "doc_id" in turn
        assert "ord" in turn
        assert "speaker" in turn
        assert "full_text" in turn
        assert "start_time_seconds" in turn
        assert "section_title" in turn
        assert "token_count" in turn

        # Verify specific values
        assert turn["doc_id"] == sample_podcast_with_turns["doc_id"]
        assert turn["ord"] == 1
        assert turn["start_time_seconds"] == 60
        assert turn["section_title"] == "AI Safety"

    def test_expand_all_chunks(self, test_client, sample_podcast_with_turns):
        """Test expanding all chunks returns all unique turns."""
        # Arrange
        chunk_ids = sample_podcast_with_turns["chunk_ids"]

        # Act
        response = test_client.post("/api/retrieval/expand", json={"chunk_ids": chunk_ids})

        # Assert
        assert response.status_code == 200
        data = response.json()

        # We have 8 chunks mapping to 6 turns
        assert data["total_turns"] == 6
        assert len(data["turns"]) == 6

        # Verify ordering by ord
        ords = [t["ord"] for t in data["turns"]]
        assert ords == [0, 1, 2, 3, 4, 5]


@pytest.mark.integration
class TestQAPairsEndpoint:
    """Test suite for POST /api/retrieval/qa-pairs endpoint."""

    def test_qa_pairs_single_turn(self, test_client, sample_podcast_with_turns):
        """Test generating Q&A pair for a single turn."""
        # Arrange - turn_ids[1] (Guest's answer) should pair with turn_ids[0] (Dwarkesh's question)
        turn_ids = [sample_podcast_with_turns["turn_ids"][1]]

        # Act
        response = test_client.post("/api/retrieval/qa-pairs", json={"turn_ids": turn_ids})

        # Assert
        assert response.status_code == 200
        data = response.json()

        assert data["total_pairs"] == 1
        assert data["skipped_turns"] == 0
        assert data["not_found_count"] == 0
        assert data["query_time_ms"] >= 0

        pair = data["pairs"][0]
        assert pair["question_turn"]["speaker"] == "Dwarkesh Patel"
        assert pair["question_turn"]["full_text"] == "What are your thoughts on AI safety?"
        assert pair["answer_turn"]["speaker"] == "Test Guest"
        assert "AI safety is crucial" in pair["answer_turn"]["full_text"]

    def test_qa_pairs_multiple_turns(self, test_client, sample_podcast_with_turns):
        """Test generating Q&A pairs for multiple turns."""
        # Arrange - turns 1, 3, 5 (guest answers)
        turn_ids = [
            sample_podcast_with_turns["turn_ids"][1],
            sample_podcast_with_turns["turn_ids"][3],
            sample_podcast_with_turns["turn_ids"][5],
        ]

        # Act
        response = test_client.post("/api/retrieval/qa-pairs", json={"turn_ids": turn_ids})

        # Assert
        assert response.status_code == 200
        data = response.json()

        assert data["total_pairs"] == 3
        assert data["skipped_turns"] == 0
        assert data["not_found_count"] == 0

        # All pairs should have Dwarkesh asking, Guest answering
        for pair in data["pairs"]:
            assert pair["question_turn"]["speaker"] == "Dwarkesh Patel"
            assert pair["answer_turn"]["speaker"] == "Test Guest"

    def test_qa_pairs_first_turn_skipped(self, test_client, sample_podcast_with_turns):
        """Test that first turn in document is skipped (no previous turn)."""
        # Arrange - turn_ids[0] is the first turn (ord=0)
        turn_ids = [sample_podcast_with_turns["turn_ids"][0]]

        # Act
        response = test_client.post("/api/retrieval/qa-pairs", json={"turn_ids": turn_ids})

        # Assert
        assert response.status_code == 200
        data = response.json()

        assert data["total_pairs"] == 0
        assert data["skipped_turns"] == 1  # First turn has no previous
        assert data["not_found_count"] == 0

    def test_qa_pairs_mixed_valid_first_turns(self, test_client, sample_podcast_with_turns):
        """Test mix of first turn (skipped) and valid turns."""
        # Arrange
        turn_ids = [
            sample_podcast_with_turns["turn_ids"][0],  # First turn - will be skipped
            sample_podcast_with_turns["turn_ids"][1],  # Valid - has previous
        ]

        # Act
        response = test_client.post("/api/retrieval/qa-pairs", json={"turn_ids": turn_ids})

        # Assert
        assert response.status_code == 200
        data = response.json()

        assert data["total_pairs"] == 1
        assert data["skipped_turns"] == 1
        assert data["not_found_count"] == 0

    def test_qa_pairs_nonexistent_turn_ids(self, test_client, sample_podcast_with_turns):
        """Test Q&A pairs with non-existent turn IDs."""
        # Arrange
        turn_ids = [999999, 888888]

        # Act
        response = test_client.post("/api/retrieval/qa-pairs", json={"turn_ids": turn_ids})

        # Assert
        assert response.status_code == 200
        data = response.json()

        assert data["total_pairs"] == 0
        assert data["skipped_turns"] == 0
        assert data["not_found_count"] == 2  # Both IDs not found

    def test_qa_pairs_mixed_valid_invalid_turn_ids(self, test_client, sample_podcast_with_turns):
        """Test Q&A pairs with mix of valid and invalid turn IDs."""
        # Arrange
        turn_ids = [
            sample_podcast_with_turns["turn_ids"][1],  # Valid
            999999,  # Invalid
        ]

        # Act
        response = test_client.post("/api/retrieval/qa-pairs", json={"turn_ids": turn_ids})

        # Assert
        assert response.status_code == 200
        data = response.json()

        assert data["total_pairs"] == 1
        assert data["skipped_turns"] == 0
        assert data["not_found_count"] == 1

    def test_qa_pairs_duplicate_turn_ids_deduplicated(self, test_client, sample_podcast_with_turns):
        """Test that duplicate turn IDs are deduplicated."""
        # Arrange - same turn ID three times
        turn_ids = [
            sample_podcast_with_turns["turn_ids"][1],
            sample_podcast_with_turns["turn_ids"][1],
            sample_podcast_with_turns["turn_ids"][1],
        ]

        # Act
        response = test_client.post("/api/retrieval/qa-pairs", json={"turn_ids": turn_ids})

        # Assert
        assert response.status_code == 200
        data = response.json()

        # Should only return 1 pair (deduplicated)
        assert data["total_pairs"] == 1
        assert data["skipped_turns"] == 0
        assert data["not_found_count"] == 0

    def test_qa_pairs_empty_turn_ids_rejected(self, test_client, sample_podcast_with_turns):
        """Test that empty turn_ids list is rejected by validation."""
        # Arrange
        turn_ids: list[int] = []

        # Act
        response = test_client.post("/api/retrieval/qa-pairs", json={"turn_ids": turn_ids})

        # Assert
        assert response.status_code == 422  # Pydantic validation error

    def test_qa_pairs_returns_correct_fields(self, test_client, sample_podcast_with_turns):
        """Test that all Q&A pair fields are returned correctly."""
        # Arrange
        turn_ids = [sample_podcast_with_turns["turn_ids"][1]]

        # Act
        response = test_client.post("/api/retrieval/qa-pairs", json={"turn_ids": turn_ids})

        # Assert
        assert response.status_code == 200
        data = response.json()

        pair = data["pairs"][0]

        # Check question_turn fields
        q = pair["question_turn"]
        assert "turn_id" in q
        assert "doc_id" in q
        assert "ord" in q
        assert "speaker" in q
        assert "full_text" in q
        assert "start_time_seconds" in q
        assert "section_title" in q
        assert "token_count" in q

        # Check answer_turn fields
        a = pair["answer_turn"]
        assert "turn_id" in a
        assert "doc_id" in a
        assert "ord" in a
        assert "speaker" in a
        assert "full_text" in a
        assert "start_time_seconds" in a
        assert "section_title" in a
        assert "token_count" in a

        # Verify ordering (question should have ord one less than answer)
        assert q["ord"] == a["ord"] - 1

    def test_qa_pairs_flexible_speaker_pairing(self, test_client, sample_podcast_with_turns):
        """Test that Q&A pairs work with any speaker combination."""
        # Arrange - turn_ids[2] is Dwarkesh's follow-up, paired with turn_ids[1] (Guest)
        turn_ids = [sample_podcast_with_turns["turn_ids"][2]]

        # Act
        response = test_client.post("/api/retrieval/qa-pairs", json={"turn_ids": turn_ids})

        # Assert
        assert response.status_code == 200
        data = response.json()

        assert data["total_pairs"] == 1
        pair = data["pairs"][0]

        # Guest is the "question" (previous turn), Dwarkesh is the "answer" (target turn)
        assert pair["question_turn"]["speaker"] == "Test Guest"
        assert pair["answer_turn"]["speaker"] == "Dwarkesh Patel"

    def test_qa_pairs_all_guest_turns(self, test_client, sample_podcast_with_turns):
        """Test Q&A pairs for all guest turns (typical use case)."""
        # Arrange - all guest turns (1, 3, 5)
        turn_ids = [
            sample_podcast_with_turns["turn_ids"][1],
            sample_podcast_with_turns["turn_ids"][3],
            sample_podcast_with_turns["turn_ids"][5],
        ]

        # Act
        response = test_client.post("/api/retrieval/qa-pairs", json={"turn_ids": turn_ids})

        # Assert
        assert response.status_code == 200
        data = response.json()

        assert data["total_pairs"] == 3

        # Each pair should have Dwarkesh's question followed by Guest's answer
        for pair in data["pairs"]:
            assert pair["question_turn"]["speaker"] == "Dwarkesh Patel"
            assert pair["answer_turn"]["speaker"] == "Test Guest"


@pytest.mark.integration
class TestExpandAndQAPairsIntegration:
    """Test integration between expand and qa-pairs endpoints."""

    def test_expand_then_qa_pairs_workflow(self, test_client, sample_podcast_with_turns):
        """Test typical workflow: retrieve chunks, expand to turns, then get Q&A pairs."""
        # Step 1: Expand chunks to turns
        chunk_ids = [
            sample_podcast_with_turns["chunk_ids"][1],  # Guest's first chunk
            sample_podcast_with_turns["chunk_ids"][4],  # Guest's second answer chunk
        ]

        expand_response = test_client.post("/api/retrieval/expand", json={"chunk_ids": chunk_ids})
        assert expand_response.status_code == 200
        expand_data = expand_response.json()

        # Should get 2 unique turns
        assert expand_data["total_turns"] == 2

        # Step 2: Get Q&A pairs for those turns
        turn_ids = [t["turn_id"] for t in expand_data["turns"]]

        qa_response = test_client.post("/api/retrieval/qa-pairs", json={"turn_ids": turn_ids})
        assert qa_response.status_code == 200
        qa_data = qa_response.json()

        # Both turns have previous turns, so both should produce pairs
        assert qa_data["total_pairs"] == 2
        assert qa_data["skipped_turns"] == 0

        # Verify the Q&A context makes sense
        for pair in qa_data["pairs"]:
            # The question should be Dwarkesh's prompt
            assert pair["question_turn"]["speaker"] == "Dwarkesh Patel"
            # The answer should be the Guest's response
            assert pair["answer_turn"]["speaker"] == "Test Guest"
