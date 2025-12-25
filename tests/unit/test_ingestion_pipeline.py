"""Unit tests for IngestionPipeline."""

from unittest.mock import Mock, patch

import pytest

from src.ingestion.pipeline import IngestionPipeline

from dotenv import load_dotenv

load_dotenv()


class TestIngestionPipelineYouTubeURL:
    """Unit tests for IngestionPipeline.ingest_youtube_url method."""

    @patch("src.ingestion.pipeline.IngestionPipeline._generate_and_insert_embeddings")
    @patch("src.ingestion.pipeline.IngestionPipeline._insert_chunks")
    @patch("src.ingestion.pipeline.IngestionPipeline._insert_document")
    @patch("src.ingestion.pipeline.TokenBasedChunker.chunk")
    @patch("src.ingestion.pipeline.YouTubeTranscriptFetcher.fetch")
    def test_ingest_youtube_url_basic(
        self, mock_fetch, mock_chunk, mock_insert_doc, mock_insert_chunks, _
    ):
        """Test basic YouTube URL ingestion without embeddings."""
        # Arrange
        mock_doc = Mock()
        mock_doc.url = "https://www.youtube.com/watch?v=test123"
        mock_doc.title = "Test Video"
        mock_doc.text = "This is a test transcript about machine learning."
        mock_fetch.return_value = mock_doc

        # Mock chunks
        mock_chunk_1 = Mock()
        mock_chunk_1.token_count = 50
        mock_chunk_2 = Mock()
        mock_chunk_2.token_count = 60
        mock_chunk.return_value = [mock_chunk_1, mock_chunk_2]

        mock_insert_doc.return_value = 1
        mock_insert_chunks.return_value = [1, 2]

        pipeline = IngestionPipeline(generate_embeddings=False)

        # Act
        result = pipeline.ingest_youtube_url(mock_doc.url)

        # Assert
        assert result["doc_id"] == 1
        assert result["url"] == mock_doc.url
        assert result["title"] == "Test Video"
        assert result["chunk_count"] == 2
        assert result["total_tokens"] == 110
        assert result["ingestion_time_ms"] > 0
        assert result["embeddings_generated"] is False

        # Verify methods called
        mock_fetch.assert_called_once_with(mock_doc.url)
        mock_insert_doc.assert_called_once()
        mock_insert_chunks.assert_called_once()
        mock_chunk.assert_called_once_with(mock_doc.text)

    @patch("src.ingestion.pipeline.TokenBasedChunker.chunk")
    @patch("src.ingestion.pipeline.IngestionPipeline._insert_chunks")
    @patch("src.ingestion.pipeline.IngestionPipeline._insert_document")
    @patch("src.ingestion.pipeline.YouTubeTranscriptFetcher.fetch")
    def test_ingest_youtube_url_with_title_override(
        self, mock_fetch, mock_insert_doc, mock_insert_chunks, mock_chunk
    ):
        """Test YouTube ingestion with title override."""
        # Arrange
        mock_doc = Mock()
        mock_doc.url = "https://www.youtube.com/watch?v=test123"
        mock_doc.title = "Original Title"
        mock_doc.text = "Test transcript."
        mock_fetch.return_value = mock_doc

        mock_chunk_1 = Mock()
        mock_chunk_1.token_count = 10
        mock_chunk.return_value = [mock_chunk_1]

        mock_insert_doc.return_value = 1
        mock_insert_chunks.return_value = [1]

        pipeline = IngestionPipeline(generate_embeddings=False)
        custom_title = "Custom Override Title"

        # Act
        result = pipeline.ingest_youtube_url(mock_doc.url, title=custom_title)

        # Assert
        assert result["title"] == custom_title
        mock_insert_doc.assert_called_once()
        call_args = mock_insert_doc.call_args
        assert call_args[1]["title_override"] == custom_title

    @patch("src.ingestion.pipeline.TokenBasedChunker.chunk")
    @patch("src.ingestion.pipeline.IngestionPipeline._insert_chunks")
    @patch("src.ingestion.pipeline.IngestionPipeline._insert_document")
    @patch("src.ingestion.pipeline.YouTubeTranscriptFetcher.fetch")
    def test_ingest_youtube_url_with_metadata(
        self, mock_fetch, mock_insert_doc, mock_insert_chunks, mock_chunk
    ):
        """Test YouTube ingestion with custom metadata."""
        # Arrange
        mock_doc = Mock()
        mock_doc.url = "https://www.youtube.com/watch?v=test123"
        mock_doc.title = "Test Video"
        mock_doc.text = "Test transcript."
        mock_fetch.return_value = mock_doc

        mock_chunk_1 = Mock()
        mock_chunk_1.token_count = 10
        mock_chunk.return_value = [mock_chunk_1]

        mock_insert_doc.return_value = 1
        mock_insert_chunks.return_value = [1]

        pipeline = IngestionPipeline(generate_embeddings=False)
        custom_metadata = {"category": "tutorial", "topic": "ml"}

        # Act
        _result = pipeline.ingest_youtube_url(mock_doc.url, metadata=custom_metadata)

        # Assert
        mock_insert_doc.assert_called_once()
        call_args = mock_insert_doc.call_args
        assert call_args[1]["custom_metadata"] == custom_metadata

    @patch("src.ingestion.pipeline.IngestionPipeline._insert_chunks")
    @patch("src.ingestion.pipeline.IngestionPipeline._insert_document")
    @patch("src.ingestion.pipeline.YouTubeTranscriptFetcher.fetch")
    def test_ingest_youtube_url_insert_document_failure(
        self, mock_fetch, mock_insert_doc, mock_insert_chunks
    ):
        """Test handling of document insert failure."""
        # Arrange
        mock_doc = Mock()
        mock_doc.url = "https://www.youtube.com/watch?v=test123"
        mock_doc.title = "Test Video"
        mock_doc.text = "Test transcript."
        mock_fetch.return_value = mock_doc

        mock_insert_doc.return_value = None  # Simulate insert failure

        pipeline = IngestionPipeline(generate_embeddings=False)

        # Act & Assert
        with pytest.raises(ValueError, match="Failed to insert document into database"):
            pipeline.ingest_youtube_url(mock_doc.url)

        mock_insert_chunks.assert_not_called()

    @patch("src.ingestion.pipeline.IngestionPipeline._insert_document")
    @patch("src.ingestion.pipeline.YouTubeTranscriptFetcher.fetch")
    def test_ingest_youtube_url_fetch_failure(self, mock_fetch, mock_insert_doc):
        """Test handling of transcript fetch failure."""
        # Arrange
        mock_fetch.side_effect = ValueError("No transcript available for video")

        pipeline = IngestionPipeline(generate_embeddings=False)

        # Act & Assert
        with pytest.raises(ValueError, match="No transcript available"):
            pipeline.ingest_youtube_url("https://www.youtube.com/watch?v=test123")

        mock_insert_doc.assert_not_called()

    @patch("src.ingestion.pipeline.EmbeddingService")
    @patch("src.ingestion.pipeline.IngestionPipeline._generate_and_insert_embeddings")
    @patch("src.ingestion.pipeline.IngestionPipeline._insert_chunks")
    @patch("src.ingestion.pipeline.IngestionPipeline._insert_document")
    @patch("src.ingestion.pipeline.YouTubeTranscriptFetcher.fetch")
    def test_ingest_youtube_url_with_embeddings(
        self,
        mock_fetch,
        mock_insert_doc,
        mock_insert_chunks,
        mock_gen_embeddings,
        mock_embedding_service_class,
    ):
        """Test YouTube ingestion with embeddings enabled."""
        # Arrange
        mock_doc = Mock()
        mock_doc.url = "https://www.youtube.com/watch?v=test123"
        mock_doc.title = "Test Video"
        mock_doc.text = "Test transcript."
        mock_fetch.return_value = mock_doc

        mock_insert_doc.return_value = 1
        mock_insert_chunks.return_value = [1, 2]

        # Mock embedding service
        mock_embedding_service = Mock()
        mock_embedding_service_class.return_value = mock_embedding_service

        pipeline = IngestionPipeline(generate_embeddings=True)

        # Act
        result = pipeline.ingest_youtube_url(mock_doc.url)

        # Assert
        assert result["embeddings_generated"] is True
        mock_gen_embeddings.assert_called_once()

    @patch("src.ingestion.pipeline.IngestionPipeline._insert_chunks")
    @patch("src.ingestion.pipeline.IngestionPipeline._insert_document")
    @patch("src.ingestion.pipeline.YouTubeTranscriptFetcher.fetch")
    def test_ingest_youtube_url_timing(
        self, mock_fetch, mock_insert_doc, mock_insert_chunks
    ):
        """Test that ingestion timing is recorded."""
        # Arrange
        mock_doc = Mock()
        mock_doc.url = "https://www.youtube.com/watch?v=test123"
        mock_doc.title = "Test Video"
        mock_doc.text = "Test transcript."
        mock_fetch.return_value = mock_doc

        mock_insert_doc.return_value = 1
        mock_insert_chunks.return_value = [1]

        pipeline = IngestionPipeline(generate_embeddings=False)

        # Act
        result = pipeline.ingest_youtube_url(mock_doc.url)

        # Assert
        assert "ingestion_time_ms" in result
        assert result["ingestion_time_ms"] >= 0
        assert isinstance(result["ingestion_time_ms"], float)

    @patch("src.ingestion.pipeline.IngestionPipeline._insert_chunks")
    @patch("src.ingestion.pipeline.IngestionPipeline._insert_document")
    @patch("src.ingestion.pipeline.TokenBasedChunker.chunk")
    @patch("src.ingestion.pipeline.YouTubeTranscriptFetcher.fetch")
    def test_ingest_youtube_url_chunking(
        self, mock_fetch, mock_chunk, mock_insert_doc, mock_insert_chunks
    ):
        """Test that text is properly chunked."""
        # Arrange
        mock_doc = Mock()
        mock_doc.url = "https://www.youtube.com/watch?v=test123"
        mock_doc.title = "Test Video"
        mock_doc.text = "This is a longer transcript that should be chunked."
        mock_fetch.return_value = mock_doc

        mock_chunk_obj_1 = Mock()
        mock_chunk_obj_1.text = "Chunk 1"
        mock_chunk_obj_1.token_count = 50
        mock_chunk_obj_1.ord = 0

        mock_chunk_obj_2 = Mock()
        mock_chunk_obj_2.text = "Chunk 2"
        mock_chunk_obj_2.token_count = 60
        mock_chunk_obj_2.ord = 1

        mock_chunk.return_value = [mock_chunk_obj_1, mock_chunk_obj_2]

        mock_insert_doc.return_value = 1
        mock_insert_chunks.return_value = [1, 2]

        pipeline = IngestionPipeline(generate_embeddings=False)

        # Act
        result = pipeline.ingest_youtube_url(mock_doc.url)

        # Assert
        mock_chunk.assert_called_once_with(mock_doc.text)
        assert result["chunk_count"] == 2
        assert result["total_tokens"] == 110

    @patch("src.ingestion.pipeline.IngestionPipeline._insert_chunks")
    @patch("src.ingestion.pipeline.IngestionPipeline._insert_document")
    @patch("src.ingestion.pipeline.YouTubeTranscriptFetcher.fetch")
    def test_ingest_youtube_url_both_title_and_metadata(
        self, mock_fetch, mock_insert_doc, mock_insert_chunks
    ):
        """Test YouTube ingestion with both title and metadata overrides."""
        # Arrange
        mock_doc = Mock()
        mock_doc.url = "https://www.youtube.com/watch?v=test123"
        mock_doc.title = "Original Title"
        mock_doc.text = "Test transcript."
        mock_fetch.return_value = mock_doc

        mock_insert_doc.return_value = 1
        mock_insert_chunks.return_value = [1]

        pipeline = IngestionPipeline(generate_embeddings=False)
        custom_title = "Custom Title"
        custom_metadata = {"category": "tutorial"}

        # Act
        result = pipeline.ingest_youtube_url(
            mock_doc.url, title=custom_title, metadata=custom_metadata
        )

        # Assert
        assert result["title"] == custom_title
        mock_insert_doc.assert_called_once()
        call_args = mock_insert_doc.call_args
        assert call_args[1]["title_override"] == custom_title
        assert call_args[1]["custom_metadata"] == custom_metadata
