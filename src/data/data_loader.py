"""Load YouTube channel videos into the database with progress tracking."""

from datetime import datetime
from tqdm import tqdm

from src.data.scraper import YouTubeChannelScraper
from src.ingestion.pipeline import IngestionPipeline


class ChannelDataLoader:
    """Loads videos from a YouTube channel into the database."""

    def __init__(self, generate_embeddings: bool = True):
        """
        Initialize the data loader.

        Args:
            generate_embeddings: Whether to generate embeddings for chunks
        """
        self.pipeline = IngestionPipeline(generate_embeddings=generate_embeddings)
        self.total_docs = 0
        self.total_chunks = 0
        self.failed_urls = []

    def load_channel(self, channel: str = "DwarkeshPatel") -> dict:
        """
        Load all videos from a YouTube channel into the database.

        Args:
            channel: YouTube channel handle (e.g., 'DwarkeshPatel')

        Returns:
            Dict with statistics about the ingestion process
        """
        start_time = datetime.now()

        # Scrape video URLs from channel
        print(f"\n📺 Scraping videos from @{channel}...")
        scraper = YouTubeChannelScraper(channel)

        try:
            video_urls = scraper.scrape_video_urls()
        except ValueError as e:
            print(f"❌ Failed to scrape channel: {e}")
            return {
                "success": False,
                "error": str(e),
                "total_docs": 0,
                "total_chunks": 0,
                "duration_seconds": 0,
            }

        print(f"✓ Found {len(video_urls)} videos")

        # Ingest videos with progress bar
        print("\n📥 Loading videos into database...")
        for video_url in tqdm(video_urls, desc="Ingesting videos", unit="video"):
            try:
                result = self.pipeline.ingest_youtube_url(video_url)
                self.total_docs += 1
                self.total_chunks += result.get("chunk_count", 0)
            except Exception as e:
                # Log failed URLs but continue with others
                self.failed_urls.append({"url": video_url, "error": str(e)})

        end_time = datetime.now()
        duration_seconds = (end_time - start_time).total_seconds()

        return {
            "success": True,
            "total_docs": self.total_docs,
            "total_chunks": self.total_chunks,
            "failed_count": len(self.failed_urls),
            "duration_seconds": round(duration_seconds, 2),
            "failed_urls": self.failed_urls,
        }
