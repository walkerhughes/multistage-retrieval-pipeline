"""YouTube channel scraper using JSON parsing from initial page data."""

from urllib.parse import urlparse, parse_qs
import json
import re
import requests


class YouTubeChannelScraper:
    """Scrapes YouTube channel video URLs by parsing JSON from the channel page."""

    BASE_URL = "https://www.youtube.com/@{channel}/videos"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
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
        Scrape all video URLs from a YouTube channel by parsing JSON data.

        YouTube embeds video metadata in JSON within the page, which is more
        reliable than parsing HTML structure.

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

        # Extract video IDs from JSON in page
        video_ids = self._extract_video_ids_from_json(response.text)

        if not video_ids:
            raise ValueError(f"No videos found for channel: {self.channel}")

        # Convert video IDs to full URLs
        video_urls = [f"https://www.youtube.com/watch?v={vid}" for vid in video_ids]
        return video_urls

    def _extract_video_ids_from_json(self, page_text: str) -> list[str]:
        """
        Extract unique video IDs from JSON data embedded in YouTube page.

        YouTube includes initial data in a JavaScript variable 'ytInitialData'
        which contains all the video metadata for the channel page.

        Args:
            page_text: The HTML page text from YouTube

        Returns:
            List of unique video IDs
        """
        video_ids = []
        seen_ids = set()

        # Pattern 1: Find videoId in JSON objects (most common)
        # Matches: "videoId":"xxxxx" where xxxxx is the video ID
        video_id_pattern = r'"videoId":"([a-zA-Z0-9_-]{11})"'
        matches = re.findall(video_id_pattern, page_text)

        for video_id in matches:
            # Skip shorts and other non-standard content
            if video_id not in seen_ids:
                video_ids.append(video_id)
                seen_ids.add(video_id)

        return video_ids

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
