"""Unit tests for YouTube channel scraper."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from lxml import html as html_parser
from lxml.etree import ParserError

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

    def test_normalize_url_relative(self):
        """Test normalizing relative URLs."""
        scraper = YouTubeChannelScraper("TestChannel")
        href = "/watch?v=dQw4w9WgXcQ"
        normalized = scraper._normalize_url(href, "https://www.youtube.com")
        assert normalized == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_normalize_url_absolute(self):
        """Test normalizing absolute URLs."""
        scraper = YouTubeChannelScraper("TestChannel")
        href = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        normalized = scraper._normalize_url(href, "https://www.youtube.com")
        assert normalized == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_normalize_url_invalid(self):
        """Test normalizing invalid URLs."""
        scraper = YouTubeChannelScraper("TestChannel")
        href = "some-random-text"
        normalized = scraper._normalize_url(href, "https://www.youtube.com")
        assert normalized is None

    def test_extract_video_urls_empty(self):
        """Test extracting video URLs from HTML with no videos."""
        scraper = YouTubeChannelScraper("TestChannel")
        # Create a minimal HTML tree with no video links
        html_content = "<html><body></body></html>"
        tree = html_parser.fromstring(html_content)
        urls = scraper._extract_video_urls(tree)
        assert urls == []

    def test_extract_video_urls_single_video(self):
        """Test extracting a single video URL."""
        scraper = YouTubeChannelScraper("TestChannel")
        html_content = """
            <html>
                <body>
                    <a href="/watch?v=video1">Video 1</a>
                </body>
            </html>
        """
        tree = html_parser.fromstring(html_content)
        urls = scraper._extract_video_urls(tree)
        assert len(urls) == 1
        assert "watch?v=video1" in urls[0]

    def test_extract_video_urls_multiple_videos(self):
        """Test extracting multiple video URLs."""
        scraper = YouTubeChannelScraper("TestChannel")
        html_content = """
            <html>
                <body>
                    <a href="/watch?v=video1">Video 1</a>
                    <a href="/watch?v=video2">Video 2</a>
                    <a href="/watch?v=video3">Video 3</a>
                </body>
            </html>
        """
        tree = html_parser.fromstring(html_content)
        urls = scraper._extract_video_urls(tree)
        assert len(urls) == 3

    def test_extract_video_urls_removes_duplicates(self):
        """Test that duplicate video URLs are removed."""
        scraper = YouTubeChannelScraper("TestChannel")
        html_content = """
            <html>
                <body>
                    <a href="/watch?v=video1">Thumbnail</a>
                    <a href="/watch?v=video1">Title</a>
                    <a href="/watch?v=video2">Video 2</a>
                </body>
            </html>
        """
        tree = html_parser.fromstring(html_content)
        urls = scraper._extract_video_urls(tree)
        # Should have 2 unique videos, not 3
        assert len(urls) == 2

    @patch("src.data.scraper.requests.get")
    def test_scrape_video_urls_success(self, mock_get):
        """Test successful scraping of video URLs."""
        scraper = YouTubeChannelScraper("TestChannel")

        # Mock response
        mock_response = Mock()
        mock_response.content = b"""
            <html>
                <body>
                    <a href="/watch?v=video1">Video 1</a>
                    <a href="/watch?v=video2">Video 2</a>
                </body>
            </html>
        """
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        urls = scraper.scrape_video_urls()
        assert len(urls) == 2
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

    @patch("src.data.scraper.html_parser.fromstring")
    @patch("src.data.scraper.requests.get")
    def test_scrape_video_urls_parse_error(self, mock_get, mock_parse):
        """Test handling of parse errors during scraping."""
        scraper = YouTubeChannelScraper("TestChannel")

        # Mock response
        mock_response = Mock()
        mock_response.content = b"<html></html>"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        # Mock parse to raise ParserError
        mock_parse.side_effect = ParserError("Parse error")

        with pytest.raises(ValueError, match="Failed to parse HTML"):
            scraper.scrape_video_urls()

    @patch("src.data.scraper.requests.get")
    def test_scrape_video_urls_no_videos_found(self, mock_get):
        """Test handling when no videos are found."""
        scraper = YouTubeChannelScraper("TestChannel")

        # Mock response with no video links
        mock_response = Mock()
        mock_response.content = b"<html><body>No videos</body></html>"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        with pytest.raises(ValueError, match="No videos found"):
            scraper.scrape_video_urls()

    @patch("src.data.scraper.requests.get")
    def test_scrape_video_urls_uses_headers(self, mock_get):
        """Test that scraper uses proper headers."""
        scraper = YouTubeChannelScraper("TestChannel")

        mock_response = Mock()
        mock_response.content = b'<html><body><a href="/watch?v=video1">Video</a></body></html>'
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
