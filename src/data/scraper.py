"""YouTube channel scraper using XML/HTML parsing."""

from urllib.parse import urlparse, parse_qs
from lxml import html as html_parser
from lxml.etree import ParserError
import requests


class YouTubeChannelScraper:
    """Scrapes YouTube channel video URLs using XML parsing."""

    BASE_URL = "https://www.youtube.com/@{channel}/videos"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    def __init__(self, channel: str):
        """
        Initialize scraper for a YouTube channel.

        Args:
            channel: YouTube channel handle (e.g., 'DwarkeshPatel')
        """
        self.channel = channel
        self.url = self.BASE_URL.format(channel=channel)

    def scrape_video_urls(self) -> list[str]:
        """
        Scrape all video URLs from a YouTube channel using XML parsing.

        Returns:
            List of video URLs

        Raises:
            ValueError: If unable to fetch or parse the channel page
        """
        try:
            response = requests.get(self.url, headers=self.HEADERS, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            raise ValueError(f"Failed to fetch channel page: {e}")

        try:
            tree = html_parser.fromstring(response.content)
        except ParserError as e:
            raise ValueError(f"Failed to parse HTML: {e}")

        video_urls = self._extract_video_urls(tree)

        if not video_urls:
            raise ValueError(f"No videos found for channel: {self.channel}")

        return video_urls

    def _extract_video_urls(self, tree) -> list[str]:
        """
        Extract video URLs from parsed HTML tree using XPath.

        Args:
            tree: Parsed HTML tree from lxml

        Returns:
            List of video URLs
        """
        video_urls = []
        base_youtube_url = "https://www.youtube.com"

        # Find all anchor tags that link to videos (typically in video thumbnails)
        # YouTube stores video links in thumbnail anchors with /watch?v= href
        anchor_tags = tree.xpath("//a[@href and contains(@href, '/watch?v=')]")

        for anchor in anchor_tags:
            href = anchor.get("href")
            if href:
                # Remove duplicates and normalize URLs
                video_url = self._normalize_url(href, base_youtube_url)
                if video_url and video_url not in video_urls:
                    video_urls.append(video_url)

        return video_urls

    def _normalize_url(self, href: str, base_url: str) -> str | None:
        """
        Normalize a relative or absolute video URL.

        Args:
            href: The href value from an anchor tag
            base_url: Base YouTube URL

        Returns:
            Normalized full URL or None if invalid
        """
        # Handle relative URLs
        if href.startswith("/watch"):
            return f"{base_url}{href}"

        # Handle absolute URLs
        if href.startswith("http"):
            return href

        # Invalid URL
        return None

    @staticmethod
    def extract_video_id(url: str) -> str | None:
        """
        Extract video ID from a YouTube URL.

        Args:
            url: YouTube video URL

        Returns:
            Video ID or None if unable to extract
        """
        try:
            parsed_url = urlparse(url)
            if parsed_url.netloc in ("youtube.com", "www.youtube.com"):
                query_params = parse_qs(parsed_url.query)
                if "v" in query_params:
                    return query_params["v"][0]
            elif parsed_url.netloc in ("youtu.be", "www.youtu.be"):
                # Short URL format
                return parsed_url.path.lstrip("/")
        except Exception:
            pass

        return None
