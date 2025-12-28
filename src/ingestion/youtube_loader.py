from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from langchain_community.document_loaders import YoutubeLoader

try:
    from src.ingestion.text_cleaner import clean_transcript_text
except ImportError:
    from ingestion.text_cleaner import clean_transcript_text


@dataclass
class YouTubeDocument:
    url: str
    title: str
    text: str
    published_at: Optional[datetime]
    author: Optional[str]
    metadata: dict


class YouTubeTranscriptFetcher:
    """Fetches YouTube video transcripts using LangChain."""

    @staticmethod
    def extract_video_id(url: str) -> str:
        """Extract video ID from various YouTube URL formats."""
        # Handles: youtube.com/watch?v=X, youtu.be/X, youtube.com/embed/X
        if "youtu.be/" in url:
            return url.split("youtu.be/")[1].split("?")[0]
        elif "watch?v=" in url:
            return url.split("watch?v=")[1].split("&")[0]
        elif "embed/" in url:
            return url.split("embed/")[1].split("?")[0]
        else:
            raise ValueError(f"Invalid YouTube URL: {url}")

    def fetch(self, url: str) -> YouTubeDocument:
        """
        Fetch transcript from YouTube video (without metadata).

        Args:
            url: YouTube video URL

        Returns:
            YouTubeDocument with transcript text

        Raises:
            ValueError: If URL is invalid or transcript unavailable
        """
        video_id = self.extract_video_id(url)

        # Use LangChain's YoutubeLoader without fetching video metadata
        loader = YoutubeLoader.from_youtube_url(
            url,
            add_video_info=False,  # Skip metadata fetching (avoids pytube dependency)
            language=["en"],  # Prefer English transcripts
        )

        # Load returns list of documents (usually just one)
        documents = loader.load()

        if not documents:
            raise ValueError(f"No transcript available for video: {url}")

        doc = documents[0]

        # Clean the transcript text (remove newlines and backslashes)
        cleaned_text = clean_transcript_text(doc.page_content)

        return YouTubeDocument(
            url=url,
            title=f"YouTube Video {video_id}",  # Simple default title
            text=cleaned_text,
            published_at=None,  # No metadata available
            author=None,  # No metadata available
            metadata={"video_id": video_id},  # Minimal metadata
        )
