"""Modal app for parallel YouTube transcript ingestion.

This app orchestrates:
1. discover_urls: Scrape video URLs from a YouTube channel
2. fetch_transcript: Fetch transcripts in parallel and save to Modal Volume
3. main: Orchestrate the pipeline

Usage:
  modal run src/data/modal_ingestion.py --channel DwarkeshPatel --volume channel-transcripts-1234567890
"""

import json
import os
import sys
from datetime import datetime

import modal

app = modal.App("yt-channel-ingestion")

# Get the source path for mounting via image.add_local_dir()
src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define container images with required dependencies
scraper_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("requests")
    .add_local_dir(src_path, remote_path="/src")
)

transcript_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "langchain-community",
        "youtube-transcript-api",
        "langchain",
        "tiktoken",
    )
    .add_local_dir(src_path, remote_path="/src")
)


# ============================================================================
# Function 1: discover_urls (scraper_image)
# ============================================================================


@app.function(image=scraper_image)
def discover_urls(channel: str) -> list[str]:
    """Discover all video URLs from a YouTube channel.

    Args:
        channel: YouTube channel handle (e.g., 'DwarkeshPatel')

    Returns:
        List of video URLs

    Raises:
        ValueError: If unable to fetch or parse the channel page
    """
    sys.path.insert(0, "/src")
    from data.scraper import YouTubeChannelScraper

    print(f"[discover_urls] Scraping channel: {channel}")
    scraper = YouTubeChannelScraper(channel)
    urls = scraper.scrape_video_urls()
    print(f"[discover_urls] Found {len(urls)} videos")
    return urls


# ============================================================================
# Function 2: fetch_transcript (transcript_image)
# Uses environment variable MODAL_VOLUME_NAME to know which volume to use
# ============================================================================


@app.function(
    image=transcript_image,
    volumes={"/mnt/transcripts": modal.Volume.from_name("transcripts", create_if_missing=True)},
)
def fetch_transcript(url: str) -> dict:
    """Fetch transcript from YouTube video and save to Modal Volume.

    Args:
        url: YouTube video URL

    Returns:
        Dict with success status and video_id
    """
    sys.path.insert(0, "/src")
    from ingestion.youtube_loader import YouTubeTranscriptFetcher

    video_id = None

    try:
        print(f"[fetch_transcript] Fetching: {url}")
        fetcher = YouTubeTranscriptFetcher()
        doc = fetcher.fetch(url)
        video_id = doc.metadata.get("video_id")

        # Save to volume as JSON
        result = {
            "url": url,
            "video_id": video_id,
            "transcript": doc.text,
            "success": True,
            "timestamp": datetime.now().isoformat(),
        }

        # Write to mounted volume
        vol_path = f"/mnt/transcripts/{video_id}.json"
        with open(vol_path, "w") as f:
            json.dump(result, f)

        print(f"[fetch_transcript] Success: {video_id}")
        return {"success": True, "video_id": video_id}

    except Exception as e:
        # Fallback video_id extraction if we couldn't get it from the fetcher
        if not video_id:
            try:
                if "watch?v=" in url:
                    video_id = url.split("watch?v=")[1].split("&")[0]
                elif "youtu.be/" in url:
                    video_id = url.split("youtu.be/")[1].split("?")[0]
                else:
                    video_id = "unknown"
            except Exception:
                video_id = "unknown"

        error_result = {
            "url": url,
            "video_id": video_id,
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }

        # Write error result to volume
        vol_path = f"/mnt/transcripts/{video_id}_error.json"
        with open(vol_path, "w") as f:
            json.dump(error_result, f)

        print(f"[fetch_transcript] Error for {video_id}: {str(e)}")
        return {"success": False, "video_id": video_id, "error": str(e)}


# ============================================================================
# Main Entrypoint
# ============================================================================


@app.local_entrypoint()
def main(channel: str, volume: str):
    """Orchestrate the full pipeline.

    Args:
        channel: YouTube channel handle (e.g., 'DwarkeshPatel')
        volume: Name of Modal Volume to store transcripts (currently expects 'transcripts')
    """
    # Step 1: Discover URLs
    print(f"==> Discovering URLs for channel: {channel}")
    urls = discover_urls.remote(channel)
    print(f"Found {len(urls)} videos")

    # Step 2: Fetch transcripts in parallel
    print(f"==> Fetching transcripts in parallel")
    results = list(fetch_transcript.map(urls))

    # Step 3: Summary
    succeeded = sum(1 for r in results if r["success"])
    failed = len(results) - succeeded

    print(f"✓ Success: {succeeded}/{len(results)}")
    print(f"✗ Failed: {failed}/{len(results)}")
