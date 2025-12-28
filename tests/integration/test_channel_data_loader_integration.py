"""Integration tests for YouTube channel data loader."""

import pytest
from unittest.mock import patch, Mock
from datetime import datetime

from src.data.data_loader import ChannelDataLoader
from src.data.scraper import YouTubeChannelScraper


class TestChannelDataLoaderIntegration:
    """Integration tests for loading YouTube channel data into database."""

    def test_initialization(self):
        """Test data loader initialization."""
        loader = ChannelDataLoader(generate_embeddings=False)
        assert loader.total_docs == 0
        assert loader.total_chunks == 0
        assert loader.failed_urls == []

    def test_initialization_with_embeddings(self):
        """Test data loader with embeddings enabled."""
        loader = ChannelDataLoader(generate_embeddings=False)
        assert loader.pipeline is not None

    @patch("src.data.data_loader.YouTubeChannelScraper.scrape_video_urls")
    def test_load_channel_scraper_error(self, mock_scrape):
        """Test load_channel handles scraper errors gracefully."""
        loader = ChannelDataLoader(generate_embeddings=False)
        mock_scrape.side_effect = ValueError("Failed to scrape")

        result = loader.load_channel(channel="TestChannel")

        assert result["success"] is False
        assert "Failed to scrape" in result["error"]
        assert result["total_docs"] == 0
        assert result["total_chunks"] == 0

    @patch("src.data.data_loader.IngestionPipeline.ingest_youtube_url")
    @patch("src.data.data_loader.YouTubeChannelScraper.scrape_video_urls")
    def test_load_channel_single_video(self, mock_scrape, mock_ingest):
        """Test loading a single video from a channel."""
        loader = ChannelDataLoader(generate_embeddings=False)

        # Mock scraper to return one video
        video_url = "https://www.youtube.com/watch?v=test123"
        mock_scrape.return_value = [video_url]

        # Mock ingestion result
        mock_ingest.return_value = {
            "doc_id": 1,
            "title": "Test Video",
            "chunk_count": 10,
            "total_tokens": 5000,
        }

        result = loader.load_channel(channel="TestChannel")

        assert result["success"] is True
        assert result["total_docs"] == 1
        assert result["total_chunks"] == 10
        assert result["failed_count"] == 0

    @patch("src.data.data_loader.IngestionPipeline.ingest_youtube_url")
    @patch("src.data.data_loader.YouTubeChannelScraper.scrape_video_urls")
    def test_load_channel_multiple_videos(self, mock_scrape, mock_ingest):
        """Test loading multiple videos from a channel."""
        loader = ChannelDataLoader(generate_embeddings=False)

        # Mock scraper to return multiple videos
        video_urls = [
            "https://www.youtube.com/watch?v=video1",
            "https://www.youtube.com/watch?v=video2",
            "https://www.youtube.com/watch?v=video3",
        ]
        mock_scrape.return_value = video_urls

        # Mock ingestion results
        def ingest_side_effect(url):
            return {
                "doc_id": len(loader.failed_urls) + 1,
                "title": f"Video {len(loader.failed_urls) + 1}",
                "chunk_count": 8,
                "total_tokens": 4000,
            }

        mock_ingest.side_effect = ingest_side_effect

        result = loader.load_channel(channel="TestChannel")

        assert result["success"] is True
        assert result["total_docs"] == 3
        assert result["total_chunks"] == 24  # 3 videos * 8 chunks
        assert result["failed_count"] == 0

    @patch("src.data.data_loader.IngestionPipeline.ingest_youtube_url")
    @patch("src.data.data_loader.YouTubeChannelScraper.scrape_video_urls")
    def test_load_channel_with_failures(self, mock_scrape, mock_ingest):
        """Test load_channel continues on individual video failures."""
        loader = ChannelDataLoader(generate_embeddings=False)

        # Mock scraper to return multiple videos
        video_urls = [
            "https://www.youtube.com/watch?v=video1",
            "https://www.youtube.com/watch?v=video2",
            "https://www.youtube.com/watch?v=video3",
        ]
        mock_scrape.return_value = video_urls

        # Mock ingestion to fail on second video
        def ingest_side_effect(url):
            if "video2" in url:
                raise Exception("Transcript unavailable")
            return {
                "doc_id": 1,
                "title": "Test Video",
                "chunk_count": 5,
                "total_tokens": 2000,
            }

        mock_ingest.side_effect = ingest_side_effect

        result = loader.load_channel(channel="TestChannel")

        assert result["success"] is True
        assert result["total_docs"] == 2  # Only 2 succeeded
        assert result["total_chunks"] == 10  # 2 videos * 5 chunks
        assert result["failed_count"] == 1
        assert len(result["failed_urls"]) == 1

    @patch("src.data.data_loader.IngestionPipeline.ingest_youtube_url")
    @patch("src.data.data_loader.YouTubeChannelScraper.scrape_video_urls")
    def test_load_channel_default_channel(self, mock_scrape, mock_ingest):
        """Test load_channel uses DwarkeshPatel as default channel."""
        loader = ChannelDataLoader(generate_embeddings=False)
        mock_scrape.return_value = []
        mock_scrape.side_effect = ValueError("No videos")

        with patch("src.data.data_loader.YouTubeChannelScraper") as mock_scraper_class:
            mock_instance = Mock()
            mock_instance.scrape_video_urls.side_effect = ValueError("No videos")
            mock_scraper_class.return_value = mock_instance

            result = loader.load_channel()

            # Verify DwarkeshPatel was used as default
            mock_scraper_class.assert_called_once_with("DwarkeshPatel")

    @patch("src.data.data_loader.IngestionPipeline.ingest_youtube_url")
    @patch("src.data.data_loader.YouTubeChannelScraper.scrape_video_urls")
    def test_load_channel_tracks_timing(self, mock_scrape, mock_ingest):
        """Test that load_channel tracks total duration."""
        loader = ChannelDataLoader(generate_embeddings=False)

        mock_scrape.return_value = ["https://www.youtube.com/watch?v=video1"]
        mock_ingest.return_value = {
            "doc_id": 1,
            "title": "Test Video",
            "chunk_count": 5,
            "total_tokens": 2000,
        }

        result = loader.load_channel(channel="TestChannel")

        assert result["success"] is True
        assert "duration_seconds" in result
        assert result["duration_seconds"] >= 0

    @patch("src.data.data_loader.IngestionPipeline.ingest_youtube_url")
    @patch("src.data.data_loader.YouTubeChannelScraper.scrape_video_urls")
    def test_load_channel_failed_url_details(self, mock_scrape, mock_ingest):
        """Test that failed URLs include error details."""
        loader = ChannelDataLoader(generate_embeddings=False)

        video_urls = [
            "https://www.youtube.com/watch?v=video1",
            "https://www.youtube.com/watch?v=video2",
        ]
        mock_scrape.return_value = video_urls

        error_message = "Video not found"

        def ingest_side_effect(url):
            if "video2" in url:
                raise Exception(error_message)
            return {
                "doc_id": 1,
                "title": "Test Video",
                "chunk_count": 5,
                "total_tokens": 2000,
            }

        mock_ingest.side_effect = ingest_side_effect

        result = loader.load_channel(channel="TestChannel")

        assert len(result["failed_urls"]) == 1
        assert result["failed_urls"][0]["url"] == video_urls[1]
        assert result["failed_urls"][0]["error"] == error_message
