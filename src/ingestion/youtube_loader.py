from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from langchain_community.document_loaders import YoutubeLoader


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
        Fetch transcript and metadata from YouTube video.

        Args:
            url: YouTube video URL

        Returns:
            YouTubeDocument with transcript and metadata

        Raises:
            ValueError: If URL is invalid or transcript unavailable
        """
        video_id = self.extract_video_id(url)

        # Use LangChain's YoutubeLoader
        loader = YoutubeLoader.from_youtube_url(
            url,
            add_video_info=True,  # Fetch metadata
            language=["en"],  # Prefer English transcripts
        )

        # Load returns list of documents (usually just one)
        documents = loader.load()

        if not documents:
            raise ValueError(f"No transcript available for video: {url}")

        doc = documents[0]

        # Extract metadata
        metadata = doc.metadata

        # Parse published date if available
        published_at = None
        if "publish_date" in metadata:
            try:
                # Format: "2023-01-15 00:00:00"
                published_at = datetime.strptime(
                    metadata["publish_date"], "%Y-%m-%d %H:%M:%S"
                )
            except (ValueError, KeyError):
                pass

        return YouTubeDocument(
            url=url,
            title=metadata.get("title", "Unknown Title"),
            text=doc.page_content,
            published_at=published_at,
            author=metadata.get("author", None),
            metadata={
                "video_id": video_id,
                "duration": metadata.get("length"),
                "view_count": metadata.get("view_count"),
                "description": metadata.get("description"),
            },
        )
