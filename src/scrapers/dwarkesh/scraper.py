"""HTTP scraper for Dwarkesh Podcast episodes.

Handles:
- Episode discovery from archive page
- Individual episode fetching
- Rate limiting and retries
"""

import re
import time
from datetime import datetime

import httpx
from bs4 import BeautifulSoup

from src.scrapers.dwarkesh.models import Episode, EpisodeMetadata
from src.scrapers.dwarkesh.parser import DwarkeshParser


# Dwarkesh Podcast URLs
ARCHIVE_URL = "https://www.dwarkesh.com/podcast/archive?sort=new"
BASE_URL = "https://www.dwarkesh.com"

# HTTP settings
DEFAULT_TIMEOUT = 30.0
DEFAULT_DELAY = 1.0  # Seconds between requests


class DwarkeshScraper:
    """Scraper for Dwarkesh Podcast episodes."""

    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        delay: float = DEFAULT_DELAY,
    ):
        """
        Initialize scraper.

        Args:
            timeout: HTTP request timeout in seconds
            delay: Delay between requests in seconds (for rate limiting)
        """
        self.timeout = timeout
        self.delay = delay
        self.parser = DwarkeshParser()
        self._client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                timeout=self.timeout,
                follow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                },
            )
        return self._client

    def close(self) -> None:
        """Close HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def discover_episodes(self) -> list[EpisodeMetadata]:
        """
        Discover all episode URLs from the archive page.

        Returns:
            List of EpisodeMetadata sorted by published date (oldest first)
        """
        client = self._get_client()
        response = client.get(ARCHIVE_URL)
        response.raise_for_status()

        return self._parse_archive_page(response.text)

    def _parse_archive_page(self, html: str) -> list[EpisodeMetadata]:
        """Parse archive page HTML to extract episode metadata."""
        soup = BeautifulSoup(html, "lxml")
        episodes = []

        # Find all podcast episode links
        # Substack archive typically has links in article cards
        for link in soup.find_all("a", href=True):
            href = str(link["href"])

            # Match podcast episode URLs: /p/episode-slug or full URLs
            if "/p/" in href and not href.endswith("/comments"):
                slug = href.split("/p/")[1].split("/")[0].split("?")[0]
                url = f"{BASE_URL}/p/{slug}"

                # Skip duplicates
                if any(e.slug == slug for e in episodes):
                    continue

                # Try to get title from link text or parent
                title = link.get_text().strip()
                if not title or len(title) < 3:
                    # Try parent element
                    parent = link.find_parent(["h2", "h3", "div"])
                    if parent:
                        title = parent.get_text().strip()

                if not title:
                    title = slug.replace("-", " ").title()

                # Try to find date
                published_at = None
                date_elem = link.find_next("time") or link.find_previous("time")
                if date_elem and date_elem.get("datetime"):
                    try:
                        dt_str = str(date_elem["datetime"])
                        published_at = datetime.fromisoformat(
                            dt_str.replace("Z", "+00:00")
                        )
                    except ValueError:
                        pass

                episodes.append(
                    EpisodeMetadata(
                        url=url,
                        slug=slug,
                        title=title,
                        published_at=published_at,
                    )
                )

        # Sort by published date (oldest first for ingestion order)
        # Episodes without dates go to the end
        episodes.sort(
            key=lambda e: (e.published_at is None, e.published_at or datetime.max)
        )

        return episodes

    def fetch_episode_html(self, url: str) -> str:
        """
        Fetch raw HTML for an episode page.

        Args:
            url: Episode URL

        Returns:
            Raw HTML content
        """
        client = self._get_client()
        response = client.get(url)
        response.raise_for_status()

        if self.delay > 0:
            time.sleep(self.delay)

        return response.text

    def scrape_episode(self, url: str) -> Episode:
        """
        Scrape and parse a single episode.

        Args:
            url: Episode URL

        Returns:
            Parsed Episode with turns and sections
        """
        html = self.fetch_episode_html(url)
        return self.parser.parse_full_episode(html, url)

    def scrape_all_episodes(
        self,
        urls: list[str] | None = None,
        progress_callback=None,
    ) -> list[Episode]:
        """
        Scrape all episodes (or specified URLs).

        Args:
            urls: Optional list of URLs to scrape. If None, discovers from archive.
            progress_callback: Optional callback(current, total, episode) for progress

        Returns:
            List of parsed Episodes
        """
        if urls is None:
            metadata_list = self.discover_episodes()
            urls = [m.url for m in metadata_list]

        episodes = []
        total = len(urls)

        for i, url in enumerate(urls):
            try:
                episode = self.scrape_episode(url)
                episodes.append(episode)

                if progress_callback:
                    progress_callback(i + 1, total, episode)

            except Exception as e:
                print(f"Error scraping {url}: {e}")
                # Continue with other episodes

        return episodes


def discover_episode_urls() -> list[str]:
    """
    Convenience function to discover all episode URLs.

    Returns:
        List of episode URLs sorted oldest to newest
    """
    with DwarkeshScraper() as scraper:
        metadata = scraper.discover_episodes()
        return [m.url for m in metadata]


def scrape_single_episode(url: str) -> Episode:
    """
    Convenience function to scrape a single episode.

    Args:
        url: Episode URL

    Returns:
        Parsed Episode
    """
    with DwarkeshScraper() as scraper:
        return scraper.scrape_episode(url)
