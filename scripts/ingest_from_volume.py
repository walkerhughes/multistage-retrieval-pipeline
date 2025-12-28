"""Ingest transcripts from Modal Volume to local Postgres via API.

This script:
1. Reads JSON transcript files from a directory
2. POSTs each transcript to the local API (/api/ingest/text)
3. Tracks success/failure counts

Usage:
  python scripts/ingest_from_volume.py --dir ./tmp/transcripts
  python scripts/ingest_from_volume.py --dir ./tmp/transcripts --api-url http://localhost:8000
"""

import argparse
import json
from pathlib import Path

import requests


def ingest_transcript(json_file: Path, api_url: str) -> bool:
    """Ingest a single transcript from JSON file to local API.

    Args:
        json_file: Path to JSON file containing transcript
        api_url: Base URL of local API (e.g., http://localhost:8000)

    Returns:
        True if ingestion succeeded, False otherwise
    """
    with open(json_file) as f:
        data = json.load(f)

    # Skip error files
    if not data.get("success"):
        print(f"✗ Skipping failed: {json_file.name} ({data.get('error', 'unknown error')})")
        return False

    try:
        # POST to local API
        response = requests.post(
            f"{api_url}/api/ingest/text",
            json={
                "text": data["transcript"],
                "title": f"YouTube Video {data['video_id']}",
                "metadata": {
                    "video_id": data["video_id"],
                    "url": data["url"],
                    "fetched_at": data["timestamp"],
                },
            },
            timeout=60,
        )

        response.raise_for_status()
        print(f"✓ Ingested: {data['video_id']}")
        return True

    except requests.RequestException as e:
        print(f"✗ Failed to ingest {data['video_id']}: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest transcripts from volume directory to local API"
    )
    parser.add_argument("--dir", required=True, help="Directory containing transcript JSON files")
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Base URL of local API (default: http://localhost:8000)",
    )
    args = parser.parse_args()

    # Find all JSON files
    transcript_dir = Path(args.dir)
    json_files = sorted(transcript_dir.glob("*.json"))

    # Filter out error files (keep only successful transcripts)
    transcript_files = [f for f in json_files if not f.name.endswith("_error.json")]

    if not transcript_files:
        print(f"No transcripts found in {args.dir}")
        return

    print(f"Found {len(transcript_files)} transcripts to ingest")
    print(f"API URL: {args.api_url}")
    print()

    # Ingest each transcript
    succeeded = 0
    for json_file in transcript_files:
        if ingest_transcript(json_file, args.api_url):
            succeeded += 1

    # Summary
    failed = len(transcript_files) - succeeded
    print()
    print(f"==> Ingestion complete: {succeeded}/{len(transcript_files)} succeeded")
    if failed > 0:
        print(f"    {failed} failed (check above for details)")


if __name__ == "__main__":
    main()
