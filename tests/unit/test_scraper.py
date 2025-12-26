"""Unit tests for YouTube channel scraper."""

import pytest
from unittest.mock import Mock, patch

from src.data.scraper import YouTubeChannelScraper


class TestYouTubeChannelScraper:
    """Test suite for YouTubeChannelScraper."""

    def test_initialization(self):
        """Test scraper initialization."""
        scraper = YouTubeChannelScraper("DwarkeshPatel")
        assert scraper.channel == "DwarkeshPatel"
        assert scraper.url == "https://www.youtube.com/@DwarkeshPatel/videos"

    def test_initialization_with_different_channel(self):
        """Test scraper with different channel names."""
        scraper = YouTubeChannelScraper("TestChannel")
        assert scraper.url == "https://www.youtube.com/@TestChannel/videos"

    def test_extract_video_id_standard_url(self):
        """Test extracting video ID from standard YouTube URL."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        video_id = YouTubeChannelScraper.extract_video_id(url)
        assert video_id == "dQw4w9WgXcQ"

    def test_extract_video_id_short_url(self):
        """Test extracting video ID from shortened YouTube URL."""
        url = "https://youtu.be/dQw4w9WgXcQ"
        video_id = YouTubeChannelScraper.extract_video_id(url)
        assert video_id == "dQw4w9WgXcQ"

    def test_extract_video_id_invalid_url(self):
        """Test extracting video ID from invalid URL."""
        url = "https://example.com/video"
        video_id = YouTubeChannelScraper.extract_video_id(url)
        assert video_id is None

    def test_extract_video_id_malformed_url(self):
        """Test extracting video ID from malformed URL."""
        url = "not a valid url"
        video_id = YouTubeChannelScraper.extract_video_id(url)
        assert video_id is None

    def test_extract_video_ids_from_json_single_video(self):
        """Test extracting a single video ID from JSON."""
        scraper = YouTubeChannelScraper("TestChannel")
        page_text = 'var ytInitialData = {"videoId":"dQw4w9WgXcQ"}'
        ids = scraper._extract_video_ids_from_json(page_text)
        assert len(ids) == 1
        assert ids[0] == "dQw4w9WgXcQ"

    def test_extract_video_ids_from_json_multiple_videos(self):
        """Test extracting multiple video IDs from JSON."""
        scraper = YouTubeChannelScraper("TestChannel")
        page_text = '''
            "videoId":"video1abc1a"
            "videoId":"video2abc1b"
            "videoId":"video3abc1c"
        '''
        ids = scraper._extract_video_ids_from_json(page_text)
        assert len(ids) == 3
        assert "video1abc1a" in ids
        assert "video2abc1b" in ids
        assert "video3abc1c" in ids

    def test_extract_video_ids_from_json_removes_duplicates(self):
        """Test that duplicate video IDs are removed."""
        scraper = YouTubeChannelScraper("TestChannel")
        page_text = '''
            "videoId":"dQw4w9WgXcQ"
            "videoId":"dQw4w9WgXcQ"
            "videoId":"anotherVidX"
        '''
        ids = scraper._extract_video_ids_from_json(page_text)
        assert len(ids) == 2
        assert ids.count("dQw4w9WgXcQ") == 1
        assert "anotherVidX" in ids

    def test_extract_video_ids_from_json_empty(self):
        """Test extracting from empty page."""
        scraper = YouTubeChannelScraper("TestChannel")
        page_text = "<html></html>"
        ids = scraper._extract_video_ids_from_json(page_text)
        assert ids == []

    def test_extract_video_ids_valid_format(self):
        """Test that only valid video IDs (11 chars) are extracted."""
        scraper = YouTubeChannelScraper("TestChannel")
        page_text = '''
            "videoId":"validId1234"
            "videoId":"shortid"
            "videoId":"toolongidthatshouldnotmatch"
            "videoId":"valid1234_-"
        '''
        ids = scraper._extract_video_ids_from_json(page_text)
        # Should only match the valid ones (11 chars exactly)
        assert "validId1234" in ids
        assert "valid1234_-" in ids
        # Invalid length ones should not match
        assert "shortid" not in ids
        assert "toolongidthatshouldnotmatch" not in ids

    @patch("src.data.scraper.requests.get")
    def test_scrape_video_urls_success(self, mock_get):
        """Test successful scraping of video URLs."""
        scraper = YouTubeChannelScraper("TestChannel")

        # Mock response
        mock_response = Mock()
        mock_response.text = '''
            var ytInitialData = {"contents": [
                {"videoId":"dQw4w9WgXcQ"},
                {"videoId":"jNQXAC9IVRw"}
            ]}
        '''
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        urls = scraper.scrape_video_urls()
        assert len(urls) == 2
        assert "watch?v=dQw4w9WgXcQ" in urls[0]
        assert "watch?v=jNQXAC9IVRw" in urls[1]
        mock_get.assert_called_once()

    @patch("src.data.scraper.requests.get")
    def test_scrape_video_urls_request_error(self, mock_get):
        """Test handling of request errors during scraping."""
        scraper = YouTubeChannelScraper("TestChannel")

        # Mock request error
        import requests

        mock_get.side_effect = requests.RequestException("Connection error")

        with pytest.raises(ValueError, match="Failed to fetch channel page"):
            scraper.scrape_video_urls()

    @patch("src.data.scraper.requests.get")
    def test_scrape_video_urls_no_videos_found(self, mock_get):
        """Test handling when no videos are found."""
        scraper = YouTubeChannelScraper("TestChannel")

        # Mock response with no video data
        mock_response = Mock()
        mock_response.text = "<html>No videos</html>"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        with pytest.raises(ValueError, match="No videos found"):
            scraper.scrape_video_urls()

    @patch("src.data.scraper.requests.get")
    def test_scrape_video_urls_uses_headers(self, mock_get):
        """Test that scraper uses proper headers."""
        scraper = YouTubeChannelScraper("TestChannel")

        mock_response = Mock()
        mock_response.text = '"videoId":"dQw4w9WgXcQ"'
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        scraper.scrape_video_urls()

        # Verify headers were passed
        call_kwargs = mock_get.call_args[1]
        assert "headers" in call_kwargs
        assert "User-Agent" in call_kwargs["headers"]

    @patch("src.data.scraper.requests.get")
    def test_scrape_video_urls_timeout(self, mock_get):
        """Test handling of request timeout."""
        scraper = YouTubeChannelScraper("TestChannel")

        import requests

        mock_get.side_effect = requests.Timeout("Request timed out")

        with pytest.raises(ValueError, match="Failed to fetch channel page"):
            scraper.scrape_video_urls()

    @patch("src.data.scraper.requests.get")
    def test_scrape_video_urls_creates_valid_urls(self, mock_get):
        """Test that returned URLs are valid YouTube watch URLs."""
        scraper = YouTubeChannelScraper("TestChannel")

        mock_response = Mock()
        mock_response.text = '''
            "videoId":"abc123456XY"
            "videoId":"def123456XY"
        '''
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        urls = scraper.scrape_video_urls()

        for url in urls:
            assert url.startswith("https://www.youtube.com/watch?v=")
            assert len(url) == len("https://www.youtube.com/watch?v=") + 11

    def test_scrape_video_urls_integration_with_extract_video_id(self):
        """Test that extracted URLs work with video ID extraction."""
        scraper = YouTubeChannelScraper("TestChannel")

        # Simulate what scrape_video_urls returns
        video_ids = ["dQw4w9WgXcQ", "jNQXAC9IVRw"]
        video_urls = [f"https://www.youtube.com/watch?v={vid}" for vid in video_ids]

        # Verify we can extract IDs back
        for url in video_urls:
            extracted_id = YouTubeChannelScraper.extract_video_id(url)
            assert extracted_id in video_ids
